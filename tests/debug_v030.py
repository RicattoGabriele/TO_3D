import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from physics_engine.simp_engine_3d import SIMPOptimizer3D

opt = SIMPOptimizer3D(nelx=8, nely=8, nelz=4, volfrac=0.30, penal=3.0, rmin=1.5, solver_type='direct')
opt.fix_face('left')
opt.add_load(8, 4, 2, fz=-100.0)
opt.add_passive_box(xmin=0.0, xmax=2.0, ymin=2.0, ymax=6.0, zmin=1.0, zmax=3.0)

res = opt.solve(optimizer_type='mma', max_iter=20, tol=1e-3)
print('Iterations run:', res.iterations_run)
print('Compliance history:')
for i, c in enumerate(res.compliance_history):
    print(f'Iter {i}: C = {c:.4e}')
print('Final volume fraction:', res.volume_fraction)
