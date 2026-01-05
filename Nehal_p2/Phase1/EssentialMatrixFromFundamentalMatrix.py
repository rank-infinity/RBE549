import numpy as np  


def compute_essential_matrix(F, K1, K2=None):

    if K2 is None:
        K2 = K1
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = U @ S @ Vt

    return E