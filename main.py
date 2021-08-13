from matplotlib.pyplot import plot
from model import pca, find_k
import numpy as np
import functions


# Loading data and plotting faces
X = functions.load_data(r'D:\ML\PCA\faces.mat')
functions.plot_faces(X)

# Normalizing data
X_normalized = X / 255


# Running PCA
U, S, V = pca(X_normalized)

# Plotting first 36 eigen vectors
functions.plot_faces(np.transpose(U[:, :36]) * 255)


# Finding number of pricipal components
k, variance_retained = find_k(S)

print(f"\nnumber of principal components = {k}")


# Reduction
U_reduce = U[:, :k]
z = np.dot(X_normalized, U_reduce)


# Recovering features
X_recovered = np.dot(z, np.transpose(U_reduce))

X_recovered *= 255
functions.plot_faces(X_recovered)
