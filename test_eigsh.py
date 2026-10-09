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

sigma = 1e-6
# The prompt says wrap K - sigma * G. Let's see if sigma K - G works.
A_shift = sigma * K - G 
M_jacobi = sp.diags(1.0 / np.abs(A_shift.diagonal()))
def matvec(b):
    # solve (G - sigma K) x = b => (sigma K - G) x = -b
    x, _ = sla.cg(A_shift, -b, M=M_jacobi, rtol=1e-5)
    return x
OPinv = sla.LinearOperator(shape=(N,N), matvec=matvec, dtype=float)

try:
    # A=G, M=K. Shift-invert uses (A - sigma M) = G - sigma K.
    # OPinv computes (G - sigma K)^-1 b
    # So eigenvalues should match!
    # Because we want SA (smallest algebraic, most negative), and sigma is 1e-6.
    # Shifted eigenvalues are 1 / (lambda - sigma). 
    # Since lambda is very negative, 1/(lambda - sigma) is near 0 and negative.
    # We should use which='LM' or 'SA'? Actually shift-invert with which='LM' finds lambda closest to sigma.
    # We want lambda most negative, which means 1/(lambda - sigma) is closest to 0. 
    # So we should use which='SM' on the shifted problem! 
    # But wait, ARPACK shift-invert with which='LM' finds eigenvalues of OPinv with largest magnitude!
    # OPinv has eigenvalues 1/(lambda - sigma). Largest magnitude means lambda closest to sigma.
    # If we want most negative lambda, we want 1/(lambda - sigma) to be SMALLEST magnitude!
    # So we should use which='SM'?? 
    # Wait, 'SM' for OPinv means lambda farthest from sigma. This might not converge well.
    pass
except Exception as e:
    pass

# Actually, the user specifically says:
# "wrap K_free - sigma * G_free in a LinearOperator that solves the system iteratively ... so ARPACK doesn't attempt splu."
# If A_shift = K_free - sigma * G_free, and we solve (K_free - sigma G_free) x = b.
# Let's test if A=G, M=K, OPinv=(K - sigma G)^-1 with buckling mode works.
def matvec2(b):
    A2 = K - sigma * G
    M2 = sp.diags(1.0 / np.abs(A2.diagonal()))
    x, _ = sla.cg(A2, b, M=M2, rtol=1e-5)
    return x
OPinv2 = sla.LinearOperator(shape=(N,N), matvec=matvec2, dtype=float)
try:
    # buckling mode: A x = w M x => G x = w K x
    # wait, mode='buckling' solves K x = w G x ? No, let's see.
    pass
except Exception as e:
    pass
