import time

import numpy as np
import os
import copy
import matlab.engine
import matlab_utils
from category_gnc import solve_3dcat_with_sdp
import pnp
import sys
import cv2
import sqpnp_python

m_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(m_path, "../../lib/python"))
import bench_utils
from debug_utils import debug_print


def matlab_result2np(matlab_result):
    new_result = copy.deepcopy(matlab_result)
    new_result['weights'] = matlab_utils.matlab2np(matlab_result['weights'])
    new_result['theta_est'] = matlab_utils.matlab2np(matlab_result['theta_est'])
    new_result['R_est'] = matlab_utils.matlab2np(matlab_result['R_est'])
    new_result['t_est'] = matlab_utils.matlab2np(matlab_result['t_est'])
    new_result['c_est'] = matlab_utils.matlab2np(matlab_result['c_est'])
    new_result['residuals'] = matlab_utils.matlab2np(matlab_result['residuals'])
    new_result['detectedOutliers'] = matlab_utils.matlab2np(matlab_result['detectedOutliers'])
    return new_result


def prepare_matlab_engine(path="~/code/CertifiablyRobustPerception/CertifiablyOptimalRobustPerception/", tries=50,
                          retry_delay=1):
    """Do some preparation on the matlab engine
    """
    eng = None
    # retry
    for i in range(tries):
        try:
            eng = matlab.engine.start_matlab()
        except KeyError as e:
            if i < tries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise
        break

    eng.addpath('{}/ShapeEstimation'.format(path))
    eng.addpath('{}/utils'.format(path))
    eng.addpath('{}/ShapeEstimation/solvers'.format(path))
    eng.cd('{}/ShapeEstimation'.format(path))
    eng.addpath('{}/CategoryAPE/'.format(path))
    eng.addpath('{}/CategoryAPE/solvers'.format(path))
    eng.addpath('{}/CategoryAPE/solvers/autotune_utils'.format(path))
    eng.addpath('{}/CategoryAPE/solvers/autotune_utils/alg_utils'.format(path))
    eng.addpath('{}/CategoryAPE/solvers/autotune_utils/gnc_utils'.format(path))
    eng.addpath('{}/CategoryAPE/solvers/autotune_utils/utils'.format(path))
    eng.addpath('{}/CategoryAPE/solvers/autotune_utils/plots'.format(path))
    eng.addpath('{}/AbsolutePoseEstimation/solvers'.format(path))
    eng.addpath('../utils')
    eng.addpath('./solvers')
    return eng


def pnp_category_altern(unrectC_points, tgt_cad_db_array, lam=0.001, noise_bound=0.1):
    assert (unrectC_points.shape[0] == 3)

    def pnp_category_initialization():
        """Initialization routine. Call PACE* inside
        """
        # sdp solver inputs dimension: 79 * 3 * 4
        #                              (K, 3, N)
        R, p, q, _, residuals = solve_3dcat_with_sdp(unrectC_points, tgt_cad_db_array, lam=lam, weights=None,
                                                     enforce_csum=False, normalize_lam=False)
        return (R, p, q)

    # initialize
    R_init, p_init, q_init = pnp_category_initialization()
    rho_init = np.sum(q_init)
    t_init = p_init / rho_init
    c_init = q_init / rho_init

    # TODO: Alternating solver
    R = copy.deepcopy(R_init)
    t = copy.deepcopy(t_init)
    c = copy.deepcopy(c_init)
    # 2) Pose and shape optimization (with fixed inverse depths):
    # this can be done optimally via PACE?.
    # 3) Inverse depth optimization (with fixed pose and shape):
    # this can be done in closed form, noting that the problem
    # can be decoupled as N scalar optimization problems with
    # quadratic objectives.
    # Termination criteria?

    result = {'R_est': R, 't_est': t, 'c_est': c, 'itr': 1}

    return result


def pnp_category_local(unrectC_points, tgt_cad_db_array, lam=0.1, regularizer=1,
                       R_guess=None, t_guess=None, c_guess=None, weights=None,
                       engine=None):
    """ Use manopt as a local solver to solve category-level pnp"""
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]
    if weights is None:
        weights = np.ones((N, 1))
    assert weights.shape[0] == N

    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    problem = {}
    problem['type'] = 'category APE'
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['lambda'] = float(lam)
    problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
    problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)
    problem['weights'] = matlab_utils.np2matlab(weights)

    R_guess_mat = matlab_utils.np2matlab(R_guess)
    t_guess_mat = matlab_utils.np2matlab(np.reshape(t_guess, (3, 1)))
    c_guess_mat = matlab_utils.np2matlab(np.reshape(c_guess, (K, 1)))

    R_est, t_est, c_est, xcost, final_residuals = engine.local_reproj_category_ape(problem, R_guess_mat, t_guess_mat,
                                                                                   c_guess_mat,
                                                                                   regularizer, False, nargout=5)
    solution = {}
    solution['R_est'] = matlab_utils.matlab2np(R_est)
    solution['t_est'] = matlab_utils.matlab2np(t_est)
    solution['c_est'] = matlab_utils.matlab2np(c_est)
    solution['cost'] = xcost
    solution['final_residuals'] = matlab_utils.matlab2np(final_residuals)

    return solution


def pnp_category_gnc_l0(unrectI_points, unrectC_points, tgt_cad_db_array, lam=0.1, noise_bound_p2l=0.1,
                        noise_bound_angular=5, noise_bound_reproj=0.05,
                        engine=None, spotpath=None, mosekpath=None, stridepath=None, max_iter=None,
                        intrinsic_mat=None,
                        type='multi-gnc'):
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]
    # maximum gnc iterations
    if max_iter is None:
        max_iter = 50
    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    def multi_gnc_solver():
        """Multiple GNC runs on different shapes"""
        problem = {}
        problem['type'] = 'category APE'
        problem['N'] = float(N)
        problem['K'] = float(K)
        problem['lambda'] = float(lam)
        # p2l noise bound (for outlier free solver)
        problem['noiseBound'] = float(noise_bound_p2l)
        problem['noiseBoundSq'] = float(noise_bound_p2l ** 2)
        # angular noise bound (if use angular noise thresholding for GNC iterations)
        problem['noiseBoundAngular'] = float(noise_bound_angular)
        problem['noiseBoundAngularSq'] = float(noise_bound_angular ** 2)
        problem['noiseBoundReproj'] = float(noise_bound_reproj)
        problem['noiseBoundReprojSq'] = float(noise_bound_reproj ** 2)
        problem['cBound'] = float(1)
        problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
        problem['image_points'] = matlab_utils.np2matlab(unrectI_points)
        problem['x'] = matlab_utils.np2matlab(unrectC_points[:2, :])
        problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)
        problem['intrinsics'] = matlab_utils.np2matlab(intrinsic_mat)
        problem['max_iter'] = float(max_iter)

        path = {}
        path['spotpath'] = spotpath
        path['mosekpath'] = mosekpath
        path['stridepath'] = stridepath

        solution = engine.gnc_category_ape_pnp_multi(problem, path, 'denserelax', True, 'regularizer', 1)
        return solution

    def single_gnc_solver():
        """Helper function for using a single GNC iteration to solve"""
        """Multiple GNC runs on different shapes"""
        problem = {}
        problem['type'] = 'category APE'
        problem['N'] = float(N)
        problem['K'] = float(K)
        problem['lambda'] = float(lam)
        # p2l noise bound (for outlier free solver)
        problem['noiseBound'] = float(noise_bound_p2l)
        problem['noiseBoundSq'] = float(noise_bound_p2l ** 2)
        # angular noise bound (if use angular noise thresholding for GNC iterations)
        problem['noiseBoundAngular'] = float(noise_bound_angular)
        problem['noiseBoundAngularSq'] = float(noise_bound_angular ** 2)
        problem['noiseBoundReproj'] = float(noise_bound_reproj)
        problem['noiseBoundReprojSq'] = float(noise_bound_reproj ** 2)
        problem['cBound'] = float(1)
        problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
        problem['image_points'] = matlab_utils.np2matlab(unrectI_points)
        problem['x'] = matlab_utils.np2matlab(unrectC_points[:2, :])
        problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)
        problem['intrinsics'] = matlab_utils.np2matlab(intrinsic_mat)
        problem['max_iter'] = float(max_iter)

        path = {}
        path['spotpath'] = spotpath
        path['mosekpath'] = mosekpath
        path['stridepath'] = stridepath

        solution = engine.gnc_category_ape_pnp_single(problem, path, 'denserelax', True, 'regularizer', 1)

        return solution

    def result2np(result):
        new_result = {}
        new_result['R_est'] = matlab_utils.matlab2np(result['R_est'])
        new_result['t_est'] = matlab_utils.matlab2np(result['t_est'])
        if isinstance(result['c_est'], float):
            new_result['c_est'] = np.array([result['c_est']])
        else:
            new_result['c_est'] = matlab_utils.matlab2np(result['c_est'])
        new_result['theta_est'] = matlab_utils.matlab2np(result['theta_est'])
        new_result['residuals'] = matlab_utils.matlab2np(result['residuals'])
        new_result['detectedOutliers'] = result['detectedOutliers']
        new_result['inlier_indices'] = np.argwhere((new_result['theta_est'] > 0).flatten()).flatten()
        new_result['outlier_indices'] = np.argwhere((new_result['theta_est'] <= 0).flatten()).flatten()
        new_result['itr'] = result['itr']
        new_result['final_eta'] = result['final_eta']
        return new_result

    result = None
    if type == 'multi-gnc':
        # run an individual GNC run for each shape
        solution = multi_gnc_solver()
    elif type == 'single-gnc':
        # run an single GNC iteration, but pick the best shape based on inlier cost
        solution = single_gnc_solver()
    else:
        raise NotImplementedError

    converted_result = result2np(solution)
    return converted_result


def pnp_category_kpnp_sqpnp_opencv(kpts_I, tgt_cad_db_array):
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros((4, 1))
    bearing_vectors = kpts_I / np.linalg.norm(kpts_I, axis=0)

    all_est_poses = []
    for k in range(K):
        c_shape = tgt_cad_db_array[:, :, k]

        # SQPNP flag = 8
        # see https://docs.opencv.org/4.x/d2/d28/calib3d_8hpp.html
        success, R_vec, t = cv2.solvePnP(c_shape.T.astype('float64'), kpts_I[:2, :].T.astype('float32'),
                                         camera_matrix,
                                         dist_coeffs, flags=8)
        if t is None:
            continue
        R, _ = cv2.Rodrigues(R_vec)
        t = t.flatten()
        c_cost = bench_utils.compute_sq_p2l_cost(bearing_vectors, c_shape, R, t)
        all_est_poses.append(({"R_est": R, "t_est": t}, k, c_cost))

    # get the pnp result and reconstruct the correct shape weights
    pnp_results = sorted(all_est_poses, key=lambda x: x[-1])
    best_pnp_result, best_pnp_shape = pnp_results[0][0], pnp_results[0][1]
    best_pnp_result['c_est'] = np.zeros(K)
    best_pnp_result['c_est'][best_pnp_shape] = 1
    best_pnp_result['shape_index'] = best_pnp_shape

    return best_pnp_result


def pnp_category_kpnp_sqpnp(kpts_I, tgt_cad_db_array, weights=None, reproj_refine=True, engine=None):
    """ Run K SQPnP for category level PnP"""
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]
    bearing_vectors = kpts_I / np.linalg.norm(kpts_I, axis=0)

    if weights is None:
        weights = np.array([1.0 for _ in range(N)])

    all_est_poses = []
    for k in range(K):
        c_shape = tgt_cad_db_array[:, :, k]

        # original c++ implementation, forked to added python bindings
        sols = sqpnp_python.sqpnp_solve(projections=kpts_I[:2, :], pts_3d=c_shape,
                                        weights=weights)
        if len(sols) == 0:
            debug_print("SQPnP returns 0 solutions.")
            if reproj_refine:
                debug_print("Use refinement with identity rotation & zero translation guesses.")
                refine_sol = pnp.pnp_local(unrectC_points=kpts_I,
                                           shape=c_shape,
                                           R_guess=np.eye(3),
                                           t_guess=np.array([0, 0, 0]).reshape((3, 1)),
                                           weights=weights.reshape((N, 1)),
                                           engine=engine)
                c_cost = bench_utils.compute_reproj_cost(kpts_I, c_shape, refine_sol['R_est'], refine_sol['t_est'],
                                                         weights=weights)
                all_est_poses.append(({"R_est": refine_sol['R_est'], "t_est": refine_sol['t_est']}, k, c_cost))
        else:
            for s in sols:
                R_sqpnp, t_sqpnp = np.array(s.r_hat).reshape((3, 3)), np.array(s.t).flatten()
                if not reproj_refine:
                    c_cost = bench_utils.compute_sq_p2l_cost(bearing_vectors, c_shape, R_sqpnp, t_sqpnp,
                                                             weights=weights)
                    all_est_poses.append(({"R_est": R_sqpnp, "t_est": t_sqpnp}, k, c_cost))
                else:
                    refine_sol = pnp.pnp_local(unrectC_points=kpts_I,
                                               shape=c_shape,
                                               R_guess=R_sqpnp,
                                               t_guess=t_sqpnp.reshape((3, 1)),
                                               weights=weights.reshape((N, 1)),
                                               engine=engine)
                    c_cost = bench_utils.compute_reproj_cost(kpts_I, c_shape, refine_sol['R_est'], refine_sol['t_est'],
                                                             weights=weights)
                    all_est_poses.append(({"R_est": refine_sol['R_est'], "t_est": refine_sol['t_est']}, k, c_cost))

    # get the pnp result and reconstruct the correct shape weights
    if len(all_est_poses) > 0:
        pnp_results = sorted(all_est_poses, key=lambda x: x[-1])
        best_pnp_result, best_pnp_shape = pnp_results[0][0], pnp_results[0][1]
        best_pnp_result['c_est'] = np.zeros(K)
        best_pnp_result['c_est'][best_pnp_shape] = 1
        best_pnp_result['shape_index'] = best_pnp_shape
        return best_pnp_result
    else:
        return None


def pnp_category_gnc_ksqpnp(unrectC_points,
                            tgt_cad_db_array,
                            noise_bound_reproj=0.1,
                            max_gnc_iterations=50,
                            div_factor=1.4,
                            save_trajectory=False,
                            reproj_refine=True,
                            engine=None,
                            ):
    """ Run robust category-level PnP solver: GNC + K SQPnP"""
    N = tgt_cad_db_array.shape[1]
    weights = np.ones(N)
    stop_th = 1e-6
    barc2 = 1.0
    itr = 0
    mu = 1e-4
    sol = None

    pre_TLS_cost = np.inf
    cost_diff = np.inf

    trajectory = []
    start_time = time.time()
    while itr < max_gnc_iterations and cost_diff > stop_th:
        if np.sum(weights) < 1e-12:
            print('GNC encounters numerical issues, the solution is likely to be wrong.')
            break

        # fix weights and solve for transformation
        sol = pnp_category_kpnp_sqpnp(unrectC_points, tgt_cad_db_array, weights=weights, reproj_refine=reproj_refine,
                                      engine=engine)

        # compute new combined shape
        est_shape = np.sum(tgt_cad_db_array * sol['c_est'].flatten()[np.newaxis, np.newaxis, :], axis=-1)

        # calculate reproj residuals
        residuals = bench_utils.compute_reproj_residuals(unrectC_points,
                                                         est_shape,
                                                         sol["R_est"],
                                                         sol["t_est"])

        # fix transformations and update weights
        # print(f"weights: {weights}")
        # print(f"residuals: {residuals}")
        # print(f"R: {sol['R_est']}")
        # print(f"t: {sol['t_est']}")
        # print(f"c: {sol['c_est']}")
        # breakpoint()
        residuals = residuals / noise_bound_reproj
        residuals = residuals ** 2  # residuals normalized by noise_bound

        TLS_cost = np.inner(weights, residuals)
        cost_diff = np.abs(TLS_cost - pre_TLS_cost)

        if itr < 1:
            max_residual = np.max(residuals)
            mu = max(1 / (10 * max_residual / barc2 - 1), 1e-6)
            # mu = 1e-3
            print(f'GNC first iteration max residual: {max_residual}, set mu={mu}.')

        th1 = (mu + 1) / mu * barc2
        th2 = mu / (mu + 1) * barc2
        prev_weights = np.copy(weights)
        for i in range(N):
            if residuals[i] - th1 >= 0:
                weights[i] = 0
            elif residuals[i] - th2 <= 0:
                weights[i] = 1.0
            else:
                weights[i] = np.sqrt(
                    barc2 * mu * (mu + 1) / residuals[i]) - mu
                assert (weights[i] >= 0 and weights[i] <= 1)

        weights_diff = np.linalg.norm(weights - prev_weights)
        weights_sum = np.sum(weights)

        #  increase mu
        mu = mu * div_factor
        itr = itr + 1
        pre_TLS_cost = TLS_cost

        if save_trajectory:
            trajectory.append(sol)

    end_time = time.time()
    result = {
        "R_est": sol['R_est'],
        "t_est": sol['t_est'],
        "c_est": sol['c_est'],

        "gnc_weights": weights,
        "inlier_indices": np.argwhere((weights > 0.5).flatten()).flatten(),
        "outlier_indices": np.argwhere((weights < 0.5).flatten()).flatten(),

        "itr": itr,
        "gnc_time": end_time - start_time,
    }

    return result


def pnp_category_kpnp_outlier_free_certifiable(unrectC_points, tgt_cad_db_array, lam=0.1, noise_bound=0.1,
                                               engine=None,
                                               spotpath=None,
                                               mosekpath=None,
                                               stridepath=None,
                                               sdpt3path=None,
                                               manoptpath=None,
                                               pcrpath=None,
                                               sdpnalpath=None,
                                               yalmippath=None,
                                               inner_solver='pace',
                                               ):
    """ Run K PnP for category level PnP
    """
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]

    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    problem = {}
    problem['type'] = 'category APE'
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['lambda'] = float(lam)
    problem['noiseBound'] = float(noise_bound)
    problem['noiseBoundSq'] = float(noise_bound * noise_bound)
    problem['cBound'] = float(1)
    problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
    problem['x'] = matlab_utils.np2matlab(unrectC_points[:2, :])
    problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)

    path = {}
    path['spotpath'] = spotpath
    path['mosekpath'] = mosekpath
    path['stridepath'] = stridepath
    path['sdpt3path'] = sdpt3path
    path['manoptpath'] = manoptpath
    path['pcrpath'] = pcrpath
    path['sdpnalpath'] = sdpnalpath
    path['yalmippath'] = yalmippath

    # determine the solver arg
    if inner_solver == 'pace':
        # single shape PACE
        inner_solver_arg = 1
    elif inner_solver == 'ape-outlierfree':
        # custom implementation
        inner_solver_arg = 2
    elif inner_solver == 'ape-outlierfree-yalmip':
        # generate moment matrix using yalmip
        inner_solver_arg = 3
    elif inner_solver == 'ape-stride':
        inner_solver_arg = 5
    else:
        raise NotImplementedError

    start_time = time.time()
    R_est_matlab, t_est_matlab, c_est_matlab, metadata = engine.kpnp_category_ape(problem, path,
                                                                                  'inner_solver', inner_solver_arg,
                                                                                  'regularizer', 2,
                                                                                  nargout=4)
    end_time = time.time()

    new_result = {}
    new_result['R_est'] = matlab_utils.matlab2np(R_est_matlab)
    new_result['t_est'] = matlab_utils.matlab2np(t_est_matlab)
    if isinstance(c_est_matlab, float):
        new_result['c_est'] = np.array([c_est_matlab])
    else:
        new_result['c_est'] = matlab_utils.matlab2np(c_est_matlab)
    # new_result['residuals'] = matlab_utils.matlab2np(metadata['residuals'])
    new_result['inlier_indices'] = np.array(list(range(N)))
    new_result['outlier_indices'] = np.array([])
    new_result['eta'] = metadata['eta']
    new_result['f_est'] = metadata['f_est']
    new_result['f_sdp'] = metadata['f_sdp']
    new_result['solver_time'] = end_time - start_time
    return new_result


def pnp_category_outlier_free_certifiable(unrectC_points, tgt_cad_db_array, lam=0.1, noise_bound=0.1,
                                          regularizer=1,
                                          engine=None, spotpath=None, mosekpath=None, stridepath=None):
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]

    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    problem = {}
    problem['type'] = 'category APE'
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['lambda'] = float(lam)
    problem['noiseBound'] = float(noise_bound)
    problem['noiseBoundSq'] = float(noise_bound * noise_bound)
    problem['cBound'] = float(1)
    problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
    problem['x'] = matlab_utils.np2matlab(unrectC_points[:2, :])
    problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)

    path = {}
    path['spotpath'] = spotpath
    path['mosekpath'] = mosekpath
    path['stridepath'] = stridepath

    start_time = time.time()
    R_est_matlab, t_est_matlab, c_est_matlab, metadata = engine.outlier_free_category_ape(problem, path,
                                                                                          'denserelax', True,
                                                                                          'regularizer', regularizer,
                                                                                          nargout=4)
    end_time = time.time()

    new_result = {}
    new_result['R_est'] = matlab_utils.matlab2np(R_est_matlab)
    new_result['t_est'] = matlab_utils.matlab2np(t_est_matlab)
    if isinstance(c_est_matlab, float):
        new_result['c_est'] = np.array([c_est_matlab])
    else:
        new_result['c_est'] = matlab_utils.matlab2np(c_est_matlab)
    # new_result['residuals'] = matlab_utils.matlab2np(metadata['residuals'])
    new_result['inlier_indices'] = np.array(list(range(N)))
    new_result['outlier_indices'] = np.array([])
    new_result['eta'] = metadata['eta']
    new_result['f_est'] = metadata['f_est']
    new_result['f_sdp'] = metadata['f_sdp']
    new_result['gnc_time'] = end_time - start_time
    return new_result


def pnp_category_gnc_certifiable(unrectI_points, unrectC_points, tgt_cad_db_array, lam=0.1, noise_bound_p2l=0.1,
                                 noise_bound_angular=5,
                                 noise_bound_reproj=0.05,
                                 engine=None, spotpath=None, mosekpath=None, stridepath=None,
                                 sdpt3path=None, manoptpath=None, pcrpath=None, sdpnalpath=None, yalmippath=None,
                                 max_iter=None,
                                 intrinsic_mat=None,
                                 regularizer=0,
                                 weight_update_type='standard_p2l'):
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]

    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    problem = {}
    problem['type'] = 'category APE'
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['lambda'] = float(lam)
    # p2l noise bound (for outlier free solver)
    problem['noiseBound'] = float(noise_bound_p2l)
    problem['noiseBoundSq'] = float(noise_bound_p2l ** 2)
    # angular noise bound (if use angular noise thresholding for GNC iterations)
    problem['noiseBoundAngular'] = float(noise_bound_angular)
    problem['noiseBoundAngularSq'] = float(noise_bound_angular ** 2)
    # reproj noise bound (if use local reproj search in GNC)
    problem['noiseBoundReproj'] = float(noise_bound_reproj)
    problem['noiseBoundReprojSq'] = float(noise_bound_reproj ** 2)
    problem['cBound'] = float(1)
    problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
    problem['image_points'] = matlab_utils.np2matlab(unrectI_points)
    problem['x'] = matlab_utils.np2matlab(unrectC_points[:2, :])
    problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)
    problem['intrinsics'] = matlab_utils.np2matlab(intrinsic_mat)

    # maximum gnc iterations
    if max_iter is None:
        max_iter = 50
    problem['max_iter'] = float(max_iter)

    path = {}
    path['spotpath'] = spotpath
    path['mosekpath'] = mosekpath
    path['stridepath'] = stridepath
    if sdpt3path is not None:
        path['sdpt3path'] = sdpt3path
    if manoptpath is not None:
        path['manoptpath'] = manoptpath
    if pcrpath is not None:
        path['pcrpath'] = pcrpath
    if sdpnalpath is not None:
        path['sdpnalpath'] = sdpnalpath
    if yalmippath is not None:
        path['yalmippath'] = yalmippath

    # uncomment to save data out
    # dump_folder = os.path.join(
    #    "/home/jnshi/code/robin/experiments/category_perception/apolloscape/outputs/temp_solver_data")
    # Path(dump_folder).mkdir(parents=True, exist_ok=True)
    # scipy.io.savemat(os.path.join(dump_folder, "inst_data_1.mat"), problem)

    start_time = time.time()
    if weight_update_type == 'standard_p2l':
        # DEPRECATED: Fail under synthetic tests
        solution = engine.gnc_category_ape(problem, path, 'denserelax', True, 'regularizer', regularizer)
    elif weight_update_type == 'standard_angular':
        # DEPRECATED: Does not work well
        solution = engine.gnc_angular_category_ape(problem, path, 'denserelax', True, 'regularizer', regularizer)
    elif weight_update_type == 'standard_angular_v2':
        # DEPRECATED: Does not work well
        solution = engine.gnc_angular_v2_category_ape(problem, path, 'denserelax', True, 'regularizer', regularizer)
    elif weight_update_type == 'reproj_local_search':
        # Use this
        solution = engine.gnc_reproj_category_ape(problem, path, 'denserelax', True, 'regularizer', regularizer)
    elif weight_update_type == 'reproj_local_search_kpnp':
        solution = engine.gnc_reproj_category_kpnp_ape(problem, path, 'inner_solver', 2)
    else:
        raise ValueError("Unsupported GNC update weight type.")
    end_time = time.time()

    def result2np(result):
        new_result = {}
        new_result['R_est'] = matlab_utils.matlab2np(result['R_est'])
        new_result['t_est'] = matlab_utils.matlab2np(result['t_est'])
        if isinstance(result['c_est'], float):
            new_result['c_est'] = np.array([result['c_est']])
        else:
            new_result['c_est'] = matlab_utils.matlab2np(result['c_est'])
        new_result['theta_est'] = matlab_utils.matlab2np(result['theta_est'])
        new_result['residuals'] = matlab_utils.matlab2np(result['residuals'])
        new_result['detectedOutliers'] = result['detectedOutliers']
        new_result['inlier_indices'] = np.argwhere((new_result['theta_est'] > 0).flatten()).flatten()
        new_result['outlier_indices'] = np.argwhere((new_result['theta_est'] <= 0).flatten()).flatten()
        new_result['itr'] = result['itr']
        new_result['final_eta'] = result['final_eta']
        return new_result

    converted_result = result2np(solution)
    converted_result['gnc_time'] = end_time - start_time
    return converted_result


def pnp_category_gnc_autotune_certifiable(unrectC_points, tgt_cad_db_array, lam=0.1,
                                          engine=None, spotpath=None, mosekpath=None, stridepath=None, max_iter=None,
                                          noise_upper_bound=0.1, noise_lower_bound=0.05, regularizer=0):
    assert (noise_upper_bound >= noise_lower_bound)
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]

    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    problem = {}
    problem['type'] = 'category APE'
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['lambda'] = float(lam)
    problem['noiseBound'] = float(noise_upper_bound)
    problem['noiseBoundSq'] = float(problem['noiseBound'] * problem['noiseBound'])
    problem['noiseUpperBoundSq'] = float(noise_upper_bound * noise_upper_bound)
    problem['noiseLowerBoundSq'] = float(noise_lower_bound * noise_lower_bound)
    problem['cBound'] = float(1)
    problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
    problem['x'] = matlab_utils.np2matlab(unrectC_points[:2, :])
    problem['shapes'] = matlab_utils.np2matlab(tgt_cad_db_array)
    problem['dof'] = float(3)

    # maximum gnc iterations
    if max_iter is None:
        max_iter = 50
    problem['max_iter'] = float(max_iter)

    path = {}
    path['spotpath'] = spotpath
    path['mosekpath'] = mosekpath
    path['stridepath'] = stridepath

    # uncomment to save data out
    # dump_folder = os.path.join(
    #    "/home/jnshi/code/robin/experiments/category_perception/apolloscape/outputs/temp_solver_data")
    # Path(dump_folder).mkdir(parents=True, exist_ok=True)
    # scipy.io.savemat(os.path.join(dump_folder, "inst_data_1.mat"), problem)
    solution = engine.gnc_autotune_category_ape(problem, path, 'denserelax', True, 'regularizer', regularizer)

    def result2np(result):
        new_result = {}
        new_result['R_est'] = matlab_utils.matlab2np(result['R_est'])
        new_result['t_est'] = matlab_utils.matlab2np(result['t_est'])
        new_result['c_est'] = matlab_utils.matlab2np(result['c_est'])
        new_result['itr'] = result['itr']
        new_result['inliers'] = result['inliers']
        new_result['info'] = result['info']
        return new_result

    converted_result = result2np(solution)
    return converted_result


def solve_2d3d_catgory_gnc(tgt, cad_kpts, noise_bound, engine):
    N = tgt.shape[1]
    K = cad_kpts.shape[2]
    problem = {}
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['noiseBound'] = noise_bound
    problem['noiseBoundSq'] = noise_bound * noise_bound
    problem['cBound'] = float(1)
    problem['kpts2D'] = matlab_utils.np2matlab(tgt)
    problem['shapes'] = matlab_utils.np2matlab(cad_kpts)
    problem['R_gt'] = matlab_utils.np2matlab(np.eye(3, 3))
    problem['t_gt'] = matlab_utils.np2matlab(np.array([[0], [0]]))
    problem['c_gt'] = matlab_utils.np2matlab(np.ones((N, 1)))
    problem['translationBound'] = float(10)

    result = engine.gnc_shapeest_helper(problem)
    return matlab_result2np(result)


def solve_2d3d_catgory_outlier_free(tgt, cad_db, noise_bound, engine):
    """Run Shape* (outlier free)
    """
    N = tgt.shape[1]

    # if isinstance(cad_db, list):
    #    K = len(cad_db)
    #    assert N == cad_db[0]['kpts'].shape[1]
    #    # If cad_db is a list, then obtain keypoints from list
    #    cad_kpts = []
    #    for i in range(len(cad_db)):
    #        cad_kpts.append(cad_db[i]['kpts'])
    #    cad_kpts = np.array(cad_kpts)
    # else:  # Otherwise, cad_db is already in kpts format as a np array
    #    cad_kpts = cad_db
    #    K = cad_kpts.shape[-1]
    #    assert N == cad_kpts.shape[1]
    cad_kpts = cad_db
    K = cad_kpts.shape[-1]
    assert N == cad_kpts.shape[1]

    problem = {}
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['noiseBound'] = noise_bound
    problem['noiseBoundSq'] = noise_bound * noise_bound
    problem['cBound'] = float(1)
    problem['kpts2D'] = matlab_utils.np2matlab(tgt)
    problem['shapes'] = matlab_utils.np2matlab(cad_kpts)
    problem['R_gt'] = matlab_utils.np2matlab(np.eye(3, 3))
    problem['t_gt'] = matlab_utils.np2matlab(np.array([[0], [0]]))
    problem['c_gt'] = matlab_utils.np2matlab(np.ones((K, 1)))
    problem['translationBound'] = float(10)

    """
    problem = 
  struct with fields:

                   N: 66
                   K: 5
          noiseSigma: 0.0100
        outlierRatio: 0
    translationBound: 10
          nrOutliers: 0
          outlierIdx: []
              kpts2D: [2×66 double]
              shapes: [3×66×5 double]
                c_gt: [5×1 double]
                R_gt: [3×3 double]
                t_gt: [2×1 double]
                S_gt: [3×66 double]
          noiseBound: 0.1000
        noiseBoundSq: 0.0100
              cBound: 1
    """

    result = engine.shape_star_stride_helper(problem)

    def convert_result(matlab_result):
        new_result = copy.deepcopy(matlab_result)
        new_result['f_est'] = matlab_utils.matlab2np(matlab_result['f_est'])
        new_result['R_est'] = matlab_utils.matlab2np(matlab_result['R_est'])
        new_result['t_est'] = matlab_utils.matlab2np(matlab_result['t_est'])
        new_result['c_est'] = matlab_utils.matlab2np(matlab_result['c_est'])
        return new_result

    return convert_result(result)


def kostas_weak_persp_category_convex(tgt, tgt_cad_db_array, engine, solver_type_str='convex+refine', beta=0.1, lam=1):
    """ Use Kostas' solver to solver for 2D3D"""
    # prepare the data
    # construct the B matrix: (3*K) * N
    N = tgt.shape[1]
    K = tgt_cad_db_array.shape[2]
    assert N == tgt_cad_db_array.shape[1]

    B = np.zeros((3 * K, N))
    for k in range(K):
        B[3 * k:3 * k + 3, :] = tgt_cad_db_array[:, :, k]

    # construct and centralize basis shapes
    # find centers
    c = np.mean(B, axis=1).flatten()
    # centralize basis shapes
    B = B - c[:, np.newaxis]

    # normalize 2D coordinates
    # scale = np.mean(np.std(tgt[:2, :], axis=1, ddof=0))
    # W_in = tgt[:2, :] / scale
    W_in = tgt[:2, :]

    # call convex matlab function
    W_in_matlab = matlab_utils.np2matlab(W_in)
    B_matlab = matlab_utils.np2matlab(B)
    _, info = engine.ssr2D3D_wrapper(W_in_matlab, B_matlab, solver_type_str, 'beta', beta, 'lam', lam, 'verb', True,
                                     nargout=2)

    # extract the estimates
    # tgt =  (R_wkpersp_est @ (c_wkpersp_est * B))[:2, :] * scale + t_wkpersp_est * scale
    result = {}
    result['R_wkpersp_est'] = matlab_utils.matlab2np(info['R'])
    result['t_wkpersp_est'] = matlab_utils.matlab2np(info['T']).flatten()
    result['c_wkpersp_est'] = matlab_utils.matlab2np(info['C'])
    if type(result['c_wkpersp_est']) is float:
        result['c_wkpersp_est'] = np.array([result['c_wkpersp_est']])
    else:
        result['c_wkpersp_est'] = result['c_wkpersp_est'].flatten()

    # outlier vector: nonzero entries are outliers
    result['E_wkpersp'] = matlab_utils.matlab2np(info['E'])

    return result
