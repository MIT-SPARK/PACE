import copy
import os
import math
import autograd.numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product
from pymanopt import Problem
from pymanopt.optimizers.trust_regions import TrustRegions
import matlab.engine
import matlab_utils


def toSE3(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1
    return T


def rot2cayley(R):
    C1 = R - np.eye(3)
    C2 = R + np.eye(3)
    C = C1 @ np.linalg.inv(C2)
    v = np.zeros((3, 1))
    v[0, 0] = -C[1, 2]
    v[1, 0] = C[0, 2]
    v[2, 0] = -C[0, 1]
    return v


def cayley2rot(cayley):
    scale = 1 + math.pow(cayley[0], 2) + math.pow(cayley[1], 2) + math.pow(cayley[2], 2)
    R = np.zeros((3, 3))
    R[0, 0] = 1 + math.pow(cayley[0], 2) - math.pow(cayley[1], 2) - math.pow(cayley[2], 2)
    R[0, 1] = 2 * (cayley[0] * cayley[1] - cayley[2])
    R[0, 2] = 2 * (cayley[0] * cayley[2] + cayley[1])
    R[1, 0] = 2 * (cayley[0] * cayley[1] + cayley[2])
    R[1, 1] = 1 - math.pow(cayley[0], 2) + math.pow(cayley[1], 2) - math.pow(cayley[2], 2)
    R[1, 2] = 2 * (cayley[1] * cayley[2] - cayley[0])
    R[2, 0] = 2 * (cayley[0] * cayley[2] - cayley[1])
    R[2, 1] = 2 * (cayley[1] * cayley[2] + cayley[0])
    R[2, 2] = 1 - math.pow(cayley[0], 2) - math.pow(cayley[1], 2) + math.pow(cayley[2], 2)
    R = (1 / scale) * R
    return R


def manopt_perpare_engine(eng, solver_path="./", manopt_path=None):
    eng.addpath(os.path.abspath(solver_path))
    s = eng.genpath(manopt_path)
    eng.addpath(s, nargout=0)


def pnp_category_manopt_cost(R, t, c, unrectI_est_kpts, bearing_vectors, tgt_cad_db_array, K_intrinsic, engine, lam=0):
    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]
    problem = {}
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['R_guess'] = matlab_utils.np2matlab(R)
    problem['t_guess'] = matlab_utils.np2matlab(t.reshape(3, 1))
    problem['c_guess'] = matlab_utils.np2matlab(c.reshape(K, 1))
    problem['bearing_vectors'] = matlab_utils.np2matlab(bearing_vectors)
    problem['cad_db_array'] = matlab_utils.np2matlab(tgt_cad_db_array)
    problem['kpts_I'] = matlab_utils.np2matlab(unrectI_est_kpts[:2, :])
    problem['K_intrinsic'] = matlab_utils.np2matlab(K_intrinsic)
    problem['lambda'] = lam
    _, x_cost = engine.category_pnp(problem, True, nargout=2)
    return x_cost / float(unrectI_est_kpts.shape[1])


def pnp_category_manopt(unrectI_est_kpts, bearing_vectors, tgt_cad_db_array, t_guess, R_guess, shape_guess, K_intrinsic,
                        lam=0, mingradnorm=1e-4,
                        engine=None, solver_path="./", manopt_path="/home/jnshi/code/manopt/manopt", t_only=False,
                        fix_c=False):
    """Run category level nonlinear PnP with manopt
    """
    if engine is None:
        engine = matlab.engine.start_matlab()
        engine.addpath(os.path.abspath(solver_path))
        engine.addpath(manopt_path)

    N = tgt_cad_db_array.shape[1]
    K = tgt_cad_db_array.shape[2]
    problem = {}
    problem['N'] = float(N)
    problem['K'] = float(K)
    problem['R_guess'] = matlab_utils.np2matlab(R_guess)
    problem['t_guess'] = matlab_utils.np2matlab(t_guess.reshape(3, 1))
    problem['c_guess'] = matlab_utils.np2matlab(shape_guess.reshape(K, 1))
    problem['bearing_vectors'] = matlab_utils.np2matlab(bearing_vectors)
    problem['cad_db_array'] = matlab_utils.np2matlab(tgt_cad_db_array)
    problem['kpts_I'] = matlab_utils.np2matlab(unrectI_est_kpts)
    problem['K_intrinsic'] = matlab_utils.np2matlab(K_intrinsic)
    problem['lambda'] = lam
    if t_only:
        problem['R'] = matlab_utils.np2matlab(R_guess)
        problem['c'] = matlab_utils.np2matlab(shape_guess.reshape(K, 1))
    if fix_c:
        problem['c'] = matlab_utils.np2matlab(shape_guess.reshape(K, 1))

    def result2np(result):
        new_result = (
            matlab_utils.matlab2np(result['R']), matlab_utils.matlab2np(result['t']),
            matlab_utils.matlab2np(result['c']))
        return new_result

    if t_only:
        X, x_cost = engine.category_pnp_t_only(problem, nargout=2)
        return (R_guess, matlab_utils.matlab2np(X), shape_guess), x_cost
    elif fix_c:
        X, x_cost = engine.category_pnp_fix_c(problem, nargout=2)
        return (matlab_utils.matlab2np(X['R']), matlab_utils.matlab2np(X['t']), shape_guess), x_cost
    else:
        X, x_cost = engine.category_pnp(problem, False, nargout=2)
        return result2np(X), x_cost


def pnp_category_pymanopt(bearing_vectors, tgt_cad_db_array, t_guess, R_guess, shape_guess, lam=0, mingradnorm=1e-4):
    N = bearing_vectors.shape[1]
    K = tgt_cad_db_array.shape[0]
    manifold = Product((SpecialOrthogonalGroup(3), Euclidean(3), Euclidean(K)))

    @pymanopt.function.Autograd
    def cost(R, t, c):
        # R = X[0]
        # t = X[1]
        # c = X[2]
        # weighted points
        weighted_pts_world = np.sum(c[:, None, None] * tgt_cad_db_array, axis=0)
        weights_pts_cam = R @ weighted_pts_world + t.reshape((3, 1))
        # to rays
        weights_pts_cam_norms = np.sqrt(np.sum(weights_pts_cam * weights_pts_cam, axis=0))
        weights_pts_cam = weights_pts_cam / weights_pts_cam_norms
        # reproj errors
        reproj_errs = np.ones((N,)) - (weights_pts_cam * bearing_vectors).sum(axis=0)
        total_reproj_err = np.sum(reproj_errs)
        # regularization
        total_err = total_reproj_err + lam * np.sqrt(np.sum(c * c))
        return total_err

    problem = Problem(manifold=manifold, cost=cost, verbosity=2)
    solver = TrustRegions(mingradnorm=mingradnorm)
    Xopt = solver.solve(problem, x=[R_guess, t_guess, shape_guess])
    return Xopt


def pnp_nonlinear_category(bearing_vectors, tgt_cad_db, t_guess, R_guess, shape_guess, lam=0):
    """Perform nonlinear category pnp.
    """
    N = bearing_vectors.shape[1]
    K = len(tgt_cad_db)
    assert N == tgt_cad_db[0]['kpts'].shape[1]
    lam_sq = math.sqrt(lam)

    def get_weighted_pt(c, i):
        """Get the weighted ith point among K cad models"""
        pt_i = np.zeros((3, 1))
        for k in range(K):
            pt_i = pt_i + np.reshape(c[k] * tgt_cad_db[k]['kpts'][:, i], (3, 1))
        return pt_i

    def cost_functor(x):
        """reprojection cost function using cad db"""
        assert x.size == 6 + K
        t = x[:3]
        cayley = x[3:6]
        c = x[6:]
        c_normalized = c / np.sum(c)
        R = cayley2rot(cayley)
        T = toSE3(R, t)
        T_inv = toSE3(np.transpose(R), -np.transpose(R) @ t)
        fvec = np.zeros((N + K,))

        p_world = np.zeros((4, 1))
        p_world[-1] = 1

        # reprojection error
        for i in range(N):
            # get 3d point
            p_world[:3, 0] = get_weighted_pt(c, i).flatten()

            # compute reprojection
            p_cam = T[:3, :] @ p_world
            p_cam = p_cam / np.linalg.norm(p_cam)

            # compute score
            fvec[i] = 1 - np.transpose(p_cam) @ bearing_vectors[:, i]

        # regularization
        for k in range(K):
            fvec[N + k] = lam_sq * c[k]
        return fvec

    # formulate initial guess
    x0 = np.zeros((6 + K,))
    x0[:3] = t_guess.flatten()
    x0[3:6] = rot2cayley(R_guess).flatten()
    x0[6:] = shape_guess.flatten()
    res = least_squares(cost_functor, x0, jac='3-point')

    # extract results
    t_est = res.x[:3]
    R_est = cayley2rot(res.x[3:6])
    shape_est = res.x[6:]

    return R_est, t_est, shape_est


def gen_cad_db(N=66, K=3):
    cad_db = []
    for k in range(K):
        cad_db.append({'kpts': np.random.randn(3, N)})
    return cad_db


def cad_db_to_array(cad_db):
    K = len(cad_db)
    N = cad_db[0]['kpts'].shape[1]
    cad_db_array = np.zeros((K, 3, N))
    for k in range(K):
        cad_db_array[k, :, :] = cad_db[k]['kpts']
    return cad_db_array


def cad_db_to_array_manopt(cad_db):
    """Convert cad db to 3D array in the shape of (3, N, K) for MATLAB manopt"""
    K = len(cad_db)
    N = cad_db[0]['kpts'].shape[1]
    cad_db_array = np.zeros((3, N, K))
    for k in range(K):
        cad_db_array[:, :, k] = cad_db[k]['kpts']
    return cad_db_array


def gen_random_weighted_cad(cad_db):
    K = len(cad_db)
    N = cad_db[0]['kpts'].shape[1]
    gt_weights = np.abs(np.random.rand(K))
    gt_weights = gt_weights / np.sum(gt_weights)
    weighted_cad = np.zeros((3, N))
    for k in range(K):
        weighted_cad = weighted_cad + gt_weights[k] * cad_db[k]['kpts']
    return weighted_cad, gt_weights


def perturb(R, t, c, amplitude=0.01):
    rotation = Rotation.from_matrix(R)
    mrp = rotation.as_mrp()
    t_perturbed = copy.deepcopy(t)
    mrp_perturbed = copy.deepcopy(mrp)
    for i in range(3):
        t_perturbed[i] += (np.random.rand() - 0.5) * 2.0 * amplitude
        mrp_perturbed += (np.random.rand() - 0.5) * 2.0 * amplitude

    c_perturbed = copy.deepcopy(c)
    for i in range(c.shape[0]):
        c_perturbed[i] += (np.random.rand() - 0.5) * 2.0 * amplitude

    return Rotation.from_mrp(mrp_perturbed).as_matrix(), t_perturbed, c_perturbed


def scipy_test():
    # bearing vectors and 3D points
    N = 66
    K = 3
    cad_db = gen_cad_db(N, K)
    gt_weighted_cad_world, shape_gt = gen_random_weighted_cad(cad_db)
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    t = np.random.randn(3, 1)

    transformed_pts_cam = R @ gt_weighted_cad_world + t
    bearing_vectors = transformed_pts_cam / np.linalg.norm(transformed_pts_cam, axis=0)

    # perturb GT to get initial guesses
    R_perturbed, t_perturbed, c_perturbed = perturb(R, t, shape_gt)

    R_est, t_est, shape_est = pnp_nonlinear_category(bearing_vectors, cad_db, t, R, c_perturbed - 0.01)
    print("R_gt: \n{}".format(R))
    print("R_est: \n{}".format(R_est))
    print("t_gt: \n{}".format(t))
    print("t_est: \n{}".format(t_est))
    print("shape_gt: \n{}".format(shape_gt))
    print("shape_est: \n{}".format(shape_est / np.sum(shape_est)))
    return


def pymanopt_test():
    N = 30
    K = 4
    cad_db = gen_cad_db(N=N, K=K)
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    t = np.random.randn(3, 1)
    gt_weighted_cad_world, shape_gt = gen_random_weighted_cad(cad_db)
    cad_db_array = cad_db_to_array(cad_db)

    # generate data
    transformed_pts_cam = R @ gt_weighted_cad_world + t
    bearing_vectors = transformed_pts_cam / np.linalg.norm(transformed_pts_cam, axis=0)

    # perturb GT to get initial guesses
    R_perturbed, t_perturbed, c_perturbed = perturb(R, t, shape_gt, amplitude=0.1)

    Xopt = pnp_category_pymanopt(bearing_vectors, cad_db_array, t_perturbed.flatten(), R_perturbed, c_perturbed,
                                 lam=0.0001)
    print("R_gt: \n{}".format(R))
    print("R_est: \n{}".format(Xopt[0]))
    print("t_gt: \n{}".format(t))
    print("t_est: \n{}".format(Xopt[1]))
    print("shape_gt: \n{}".format(shape_gt))
    print("shape_est: \n{}".format(Xopt[2]))
    return


if __name__ == "__main__":
    print("Test category PnP nonlinear solver.")
    np.random.seed(0)
    N = 30
    K = 4
    cad_db = gen_cad_db(N=N, K=K)
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    t = np.random.randn(3, 1)
    gt_weighted_cad_world, shape_gt = gen_random_weighted_cad(cad_db)
    cad_db_array = cad_db_to_array_manopt(cad_db)

    # generate data
    transformed_pts_cam = R @ gt_weighted_cad_world + t
    bearing_vectors = transformed_pts_cam / np.linalg.norm(transformed_pts_cam, axis=0)

    # perturb GT to get initial guesses
    R_perturbed, t_perturbed, c_perturbed = perturb(R, t, shape_gt, amplitude=0.1)

    result = pnp_category_manopt(bearing_vectors, cad_db_array, t_perturbed.flatten(), R_perturbed, c_perturbed)
    print("R_gt: \n{}".format(R))
    print("R_est: \n{}".format(result['R_est']))
    print("t_gt: \n{}".format(t))
    print("t_est: \n{}".format(result['t_est']))
    print("shape_gt: \n{}".format(shape_gt))
    print("shape_est: \n{}".format(result['c_est']))
