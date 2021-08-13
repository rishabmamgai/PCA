from numpy.lib.twodim_base import diag
from numpy.linalg import svd
import numpy as np


def pca(X):
    m = len(X)

    covariance_mat = np.dot(np.transpose(X), X)
    sigma = covariance_mat / m

    U, S, V = svd(sigma)
    S = diag(S)

    return U, S, V


def find_k(S):
    S_total = sum(sum(S))

    k = 1
    variance_retained = 0

    while(variance_retained < 0.99):
        S_k = 0
        for i in range(k):
            S_k += S[i, i]
        
        variance_retained = S_k / S_total

        print(f"for {k} principal components, {variance_retained * 100} variance retained")
        k += 1

    return k-1, variance_retained
