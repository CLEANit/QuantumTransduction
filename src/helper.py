import numpy as np

# function to rotate 2D vectors
def rot(vec, theta):
    rot_mat = np.array([ [np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)] ])
    return rot_mat.dot(vec)

# function to find a vector orthogonal to the passed vector (2D only)
def orthogVecSlope(vec):
    A = np.array([[vec[0], vec[1]], [-vec[1], vec[0]]])
    trans_vec = np.linalg.solve(A, np.array([0., 1.]))
    if trans_vec[0] == 0.:
        a = 0.
    else:
        a = trans_vec[1] / trans_vec[0]
    return a

