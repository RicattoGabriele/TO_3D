import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

nelx, nely, nelz = 8, 8, 4
volfrac = 0.15

# What happens if eta = 0.5 vs eta = volfrac?
for it in [1, 5, 10, 15, 20]:
    beta = float(min(32.0, 1.0 + it / 10.0))
    
    # Standard implementation: eta = 0.5
    eta = 0.5
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    x_tilde = 0.15
    xPhys_fixed_eta = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
    
    # Adaptive eta: eta = volfrac
    eta_v = volfrac
    denom_v = np.tanh(beta * eta_v) + np.tanh(beta * (1.0 - eta_v))
    xPhys_adapt_eta = (np.tanh(beta * eta_v) + np.tanh(beta * (x_tilde - eta_v))) / denom_v
    
    print(f"Iter {it:2d} (beta={beta:.2f}): eta=0.5 -> xPhys={xPhys_fixed_eta:.4f} | eta=V -> xPhys={xPhys_adapt_eta:.4f}")
