import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from physics_engine.simp_engine_3d import SIMPOptimizer3D, _TOStateEvaluator

opt = SIMPOptimizer3D(nelx=4, nely=4, nelz=2, solver_type='direct')
opt.fix_face('left')
opt.add_load(4, 2, 1, fz=-50.0)
free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
evaluator = _TOStateEvaluator(optimizer=opt, free_dofs=free_dofs, force_vec=opt.force_vector, max_iter=10, tol=1e-4, opt_alpha=0.0, mode='compliance', cb=None, solver_type='direct')

x_base = np.full(opt.num_elements, 0.4, dtype=np.float64)
evaluator.evaluate(x_base)
print('Initial it:', evaluator.it)
evaluator.evaluate(x_base)
print('After identical x:', evaluator.it)
evaluator.evaluate(x_base + 1e-15)
print('After 1e-15 perturb:', evaluator.it)
evaluator.evaluate(x_base + 0.05)
print('After 0.05 perturb:', evaluator.it)
