import copy
import math
import numpy as np
import pyopengv
import cv2
import matlab.engine
import matlab_utils


def pnp_local(unrectC_points, shape,
              R_guess=None, t_guess=None, weights=None,
              engine=None):
    """ Use manopt as a local solver to solve pnp"""
    N = shape.shape[1]
    if weights is None:
        weights = np.ones((N, 1))
    assert weights.shape[0] == N

    bearing_vectors = unrectC_points / np.linalg.norm(unrectC_points, axis=0)

    problem = {}
    problem['type'] = 'APE'
    problem['N'] = float(N)
    problem['bearings'] = matlab_utils.np2matlab(bearing_vectors)
    problem['shape'] = matlab_utils.np2matlab(shape)
    problem['weights'] = matlab_utils.np2matlab(weights)

    R_guess_mat = matlab_utils.np2matlab(R_guess)
    t_guess_mat = matlab_utils.np2matlab(np.reshape(t_guess, (3, 1)))

    R_est, t_est, xcost, final_residuals = engine.local_reproj_ape(problem,
                                                                   R_guess_mat,
                                                                   t_guess_mat,
                                                                   False, nargout=4)
    solution = {}
    solution['R_est'] = matlab_utils.matlab2np(R_est)
    solution['t_est'] = matlab_utils.matlab2np(t_est)
    solution['cost'] = xcost
    solution['final_residuals'] = matlab_utils.matlab2np(final_residuals)

    return solution


def p3p_ransac(bearing_vectors, points_3d, focal_length=800, pixel_error=1):
    threshold = 1.0 - math.cos(math.atan(pixel_error / focal_length))
    result = pyopengv.absolute_pose_ransac(bearing_vectors.T, points_3d.T, "KNEIP", threshold)
    result = np.vstack((result, np.array([0, 0, 0, 1])))
    return result


def pnp_nonlinear(bearing_vectors, points_3d, t_guess, R_guess):
    """Call the OpenGV nonlinear optimization routine for PnP.
    Note: points_3d = R @ bearing_vectors + t
    """
    result = pyopengv.absolute_pose_optimize_nonlinear(bearing_vectors.T, points_3d.T, t_guess, R_guess)
    result = np.vstack((result, np.array([0, 0, 0, 1])))
    return result


def pnp_upnp(bearing_vectors, points_3d):
    result = pyopengv.absolute_pose_upnp(bearing_vectors.T, points_3d.T)
    stacked_results = [np.vstack((r, np.array([0, 0, 0, 1]))) for r in result]
    return stacked_results


def pnp_epnp(bearing_vectors, points_3d):
    result = pyopengv.absolute_pose_epnp(bearing_vectors.T, points_3d.T)
    result = np.vstack((result, np.array([0, 0, 0, 1])))
    return result


if __name__ == "__main__":
    print("Test OpenGV PnP with RANSAC")
    np.random.seed(0)

    # bearing vectors and 3D points
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    t = np.random.randn(3, 1)
    pts_cam = np.random.randn(3, 100)
    transformed_pts_world = R @ pts_cam + t
    # pts_cam = R^T * transformed_pts_world - R^T * t
    # camera optical center at (0,0)
    projected_pts = pts_cam / np.linalg.norm(pts_cam, axis=0)
    Pi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    image_points = Pi @ pts_cam
    image_points = image_points / image_points[-1, :]
    image_points = image_points[:2, :]

    # p3p
    result_p3p = p3p_ransac(projected_pts, transformed_pts_world)
    print("p3p Result R:\n{}".format(result_p3p[:3, :3]))
    print("p3p Exp    R:\n{}".format(R))
    print("p3p Result t:\n{}".format(result_p3p[:3, 3]))
    print("p3p Exp    t:\n{}".format(t))

    # upnp
    result_upnp = pnp_upnp(projected_pts, transformed_pts_world)
    print("upnp Result R:\n{}".format(result_upnp[0][:3, :3]))
    print("upnp Exp    R:\n{}".format(R))
    print("upnp Result t:\n{}".format(result_upnp[0][:3, 3]))
    print("upnp Exp    t:\n{}".format(t))

    # nonlinear
    t_guess = (copy.deepcopy(t) + 0.1).reshape(3, 1)
    R_guess = copy.deepcopy(R)
    result_nonlinear = pnp_nonlinear(projected_pts, transformed_pts_world, t_guess, R_guess)
    print("Nonlinear Result R:\n{}".format(result_nonlinear[:3, :3]))
    print("Nonlinear Exp    R:\n{}".format(R))
    print("Nonlinear Result t:\n{}".format(result_nonlinear[:3, 3]))
    print("Nonlinear Exp    t:\n{}".format(t))
