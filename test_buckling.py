import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

np.random.seed(0)
N = 100
K = sp.diags([2, -1, -1], [0, 1, -1], shape=(N, N)).tocsr()
G = sp.random(N, N, density=0.1, format='csr', data_rvs=np.random.randn)
G = G + G.T

evals1, evecs1 = sla.eigsh(G, M=K, k=3, which='SA', tol=1e-3)
print("Original evals lambda:", np.sort(evals1))
mu_orig = -evals1
print("Original mu (-lambda):", np.sort(mu_orig))

# K x = nu G x. We want nu = 1/lambda. Since lambda is very negative, nu is small negative.
# We shift by a small negative number to find them!
sigma = -1e-6
A_shift = K - sigma * G
M_jacobi = sp.diags(1.0 / np.abs(A_shift.diagonal()))
def matvec(b):
    x, _ = sla.cg(A_shift, b, M=M_jacobi, rtol=1e-5)
    return x
OPinv = sla.LinearOperator(shape=(N,N), matvec=matvec, dtype=float)

try:
    evals2, evecs2 = sla.eigsh(K, M=G, k=3, sigma=sigma, OPinv=OPinv, mode='buckling', which='LM')
    print("New method evals (nu):", np.sort(evals2))
    print("New method mu (-1/nu):", np.sort(-1.0 / evals2))
except Exception as e:
    print("Error:", e)
