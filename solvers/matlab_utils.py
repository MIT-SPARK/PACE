import numpy as np
import matlab.engine


def np2matlab(matrix):
    """
    Helper function to convert numpy matrix to matlab array
    """
    mat_matrix = matlab.double(matrix.tolist())
    return mat_matrix


def matlab2np(x):
    if type(x) is float:
        return x
    return np.array(x._data).reshape(x.size[::-1]).T
