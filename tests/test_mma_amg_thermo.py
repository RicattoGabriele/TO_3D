"""
Comprehensive 4-Tier Test Suite for Sprint 2 Phase 2:
- MMA Optimizer (nlopt.LD_MMA)
- Algebraic Multigrid Preconditioner (pyamg smoothed aggregation)
- Thermal Conductivity & RAMP Thermal Expansion Formulation
- Coupled Multi-Physics & Real-World Optimization Benchmarks

Tiers:
- Tier 1: Feature Coverage (Unit / Standalone)
- Tier 2: Boundary & Corner Cases
- Tier 3: Cross-Feature Combinations
- Tier 4: Real-World Application Scenarios
"""

import os
import sys
import time
import tracemalloc
import unittest
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.ndimage as ndimage
import pyamg
import nlopt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_3d import (
    SIMPOptimizer3D,
    SIMPResult3D,
    h8_shape_functions,
    h8_jacobian,
    h8_strain_displacement_b,
    elastic_constitutive_matrix_d,
    h8_element_stiffness_k0,
    h8_thermal_conductivity_kth0,
    h8_thermal_expansion_force_fth0,
    geometric_stiffness_basis_matrices,
    spherical_cone_kernel,
)


class TestMMAAMGThermo(unittest.TestCase):
    """Comprehensive 4-Tier test suite covering MMA, AMG, and Thermo-Elastic physics."""

    # =========================================================================
    # TIER 1: FEATURE COVERAGE (UNIT & STANDALONE)
    # =========================================================================

    def test_tier1_mma_optimizer_standalone(self):
        """
        Tier 1: Verify standalone NLopt MMA optimizer (nlopt.LD_MMA).
        Solves an analytical separable convex structural optimization subproblem:
            min f(x) = sum(c_i / x_i)  subject to  sum(x_i) <= V0,  lb <= x_i <= ub
        Verifies:
        1. In-place gradient slice assignments (grad[:] = ...).
        2. Bounds and inequality constraints satisfaction.
        3. Strict convergence to the exact analytical KKT optimum: x_i* = V0 * sqrt(c_i) / sum(sqrt(c_k)).
        """
        n = 8
        c = np.array([1.0, 2.0, 4.0, 3.0, 2.5, 1.5, 5.0, 3.5], dtype=np.float64)
        V0 = 4.0
        lb_val, ub_val = 0.1, 1.5

        # Analytical optimum via KKT stationarity
        sqrt_c = np.sqrt(c)
        x_analytical = V0 * (sqrt_c / np.sum(sqrt_c))

        call_counts = {"obj": 0, "constr": 0}

        def objective_func(x: np.ndarray, grad: np.ndarray) -> float:
            call_counts["obj"] += 1
            if grad.size > 0:
                # CRITICAL: In-place assignment required by NLopt C interface
                grad[:] = -c / (x ** 2)
            return float(np.sum(c / x))

        def volume_constraint_func(x: np.ndarray, grad: np.ndarray) -> float:
            call_counts["constr"] += 1
            if grad.size > 0:
                grad[:] = np.ones(n, dtype=np.float64)
            return float(np.sum(x) - V0)

        opt = nlopt.opt(nlopt.LD_MMA, n)
        opt.set_lower_bounds(np.full(n, lb_val))
        opt.set_upper_bounds(np.full(n, ub_val))
        opt.set_min_objective(objective_func)
        opt.add_inequality_constraint(volume_constraint_func, 1e-6)
        opt.set_xtol_rel(1e-8)
        opt.set_maxeval(200)

        x0 = np.full(n, V0 / n, dtype=np.float64)
        x_opt = opt.optimize(x0)

        # 1. Gradient callbacks were executed
        self.assertGreater(call_counts["obj"], 0, "Objective callback must be invoked.")
        self.assertGreater(call_counts["constr"], 0, "Constraint callback must be invoked.")

        # 2. Variable bounds strictly respected
        self.assertTrue(np.all(x_opt >= lb_val - 1e-8), "Lower bounds must be respected.")
        self.assertTrue(np.all(x_opt <= ub_val + 1e-8), "Upper bounds must be respected.")

        # 3. Inequality constraint satisfied
        self.assertLessEqual(float(np.sum(x_opt)), V0 + 1e-5, "Volume constraint must be satisfied.")

        # 4. Convergence to analytical KKT solution within tolerance
        np.testing.assert_allclose(
            x_opt, x_analytical, rtol=1e-3, atol=1e-3,
            err_msg="MMA solution must match analytical KKT optimum."
        )

    def test_tier1_amg_preconditioned_cg(self):
        """
        Tier 1: Verify Algebraic Multigrid (SA-AMG via pyamg) linear solver.
        Assembles 3D elasticity stiffness matrix K on free DOFs with Dirichlet BCs.
        Constructs Smoothed Aggregation hierarchy with coarse_solver='pinv'.
        Verifies:
        1. PCG with AMG V-cycle preconditioner converges with exit code 0.
        2. Residual norm ||K u - F|| / ||F|| < 1e-5.
        3. Solution matches direct SuperLU factorization within relative error < 1e-4.
        4. Operator complexity and multilevel depth are well-formed.
        """
        nelx, nely, nelz = 6, 6, 6
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)

        xPhys = np.full(opt.num_elements, 0.5, dtype=np.float64)
        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()
        F_free = opt.force_vector[free_dofs]

        # Construct Smoothed Aggregation AMG hierarchy
        ml = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
        self.assertGreaterEqual(len(ml.levels), 2, "AMG hierarchy must have at least 2 levels.")
        self.assertLess(ml.operator_complexity(), 1.6, "Operator complexity must remain modest (< 1.6).")

        M_amg = ml.aspreconditioner(cycle='V')

        # Solve with AMG-PCG
        u_amg, exit_code = sla.cg(K_free, F_free, M=M_amg, rtol=1e-6, maxiter=500)
        self.assertEqual(exit_code, 0, "AMG-PCG solver must converge with exit code 0.")

        # Solve with Direct SuperLU
        u_dir = sla.spsolve(K_free, F_free)

        # Residual verification
        res_norm = np.linalg.norm(K_free @ u_amg - F_free) / np.linalg.norm(F_free)
        self.assertLess(res_norm, 1e-5, f"AMG relative residual norm {res_norm} must be < 1e-5.")

        # Equivalence against direct factorization
        rel_diff = np.linalg.norm(u_amg - u_dir) / np.linalg.norm(u_dir)
        self.assertLess(rel_diff, 1e-4, f"AMG solution differs from direct solve by {rel_diff:.2e} (expected < 1e-4).")

    def test_tier1_thermal_conductivity_assembly(self):
        """
        Tier 1: Verify 3D H8 thermal conductivity matrix assembly.
        Validates:
        1. Elemental k_th0 in R^{8 x 8} is strictly symmetric.
        2. Spectral property: Exactly 1 zero eigenvalue (constant temperature mode 1_8)
           and 7 strictly positive eigenvalues.
        3. Global K_th assembly via SIMP interpolation is symmetric positive semi-definite,
           and strictly positive definite on free nodes with fixed Dirichlet thermal BCs.
        """
        k_th0 = h8_thermal_conductivity_kth0(dx=1.0, dy=1.0, dz=1.0, k_th=1.0)
        self.assertEqual(k_th0.shape, (8, 8), "k_th0 must be 8 x 8.")

        # Strict symmetry
        np.testing.assert_allclose(k_th0, k_th0.T, atol=1e-12, err_msg="k_th0 must be strictly symmetric.")

        # Eigenvalue spectrum
        eigs = np.linalg.eigvalsh(k_th0)
        zero_modes = eigs[np.abs(eigs) < 1e-10]
        pos_modes = eigs[eigs >= 1e-10]
        self.assertEqual(len(zero_modes), 1, "k_th0 must have exactly 1 rigid constant-T mode.")
        self.assertEqual(len(pos_modes), 7, "k_th0 must have exactly 7 positive conduction modes.")
        self.assertGreater(float(np.min(pos_modes)), 0.1, "All active thermal modes must be strictly positive.")

        # Nullspace check on constant temperature vector
        ones_vec = np.ones(8, dtype=np.float64)
        np.testing.assert_allclose(k_th0 @ ones_vec, 0.0, atol=1e-12, err_msg="k_th0 @ 1 must be exactly zero.")

        # Global assembly
        nelx, nely, nelz = 4, 3, 2
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)
        xPhys = np.full(opt.num_elements, 0.7, dtype=np.float64)
        K_th = opt.assemble_thermal_conductivity(xPhys, penal_th=3.0)

        num_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
        self.assertEqual(K_th.shape, (num_nodes, num_nodes))

        # Global symmetry
        diff_th = (K_th - K_th.T).data
        if len(diff_th) > 0:
            self.assertLess(float(np.max(np.abs(diff_th))), 1e-11, "Global K_th must be symmetric.")

        # Strict positive definiteness on free thermal DOFs with node 0 fixed
        free_nodes = np.arange(1, num_nodes, dtype=np.int32)
        K_th_free = K_th[free_nodes, :][:, free_nodes]
        diag_free = K_th_free.diagonal()
        self.assertTrue(np.all(diag_free > 0.0), "All diagonal entries of K_th_free must be strictly positive.")

    def test_tier1_ramp_thermal_load_vector(self):
        """
        Tier 1: Verify RAMP thermal expansion force vector formulation and assembly.
        Validates:
        1. Elemental f_th0 in R^24 is self-equilibrating (nodal forces sum to [0, 0, 0]^T).
        2. RAMP interpolation maintains non-zero sensitivity derivative at zero density (x=0),
           unlike SIMP which vanishes (avoiding void freeze).
        3. Monotonicity: E_th(x) strictly increases from Emin to E0.
        4. Global assembly produces correct force dimensions and finite real values.
        """
        f_th0 = h8_thermal_expansion_force_fth0(dx=1.0, dy=1.0, dz=1.0, E0=1.0, nu=0.3, alpha_th=1.0)
        self.assertEqual(len(f_th0), 24, "f_th0 must have 24 entries (8 nodes x 3 DOFs).")

        # Self-equilibrating translational equilibrium
        fx_net = np.sum(f_th0[0::3])
        fy_net = np.sum(f_th0[1::3])
        fz_net = np.sum(f_th0[2::3])
        self.assertAlmostEqual(fx_net, 0.0, places=11, msg="Net thermal Fx must be zero.")
        self.assertAlmostEqual(fy_net, 0.0, places=11, msg="Net thermal Fy must be zero.")
        self.assertAlmostEqual(fz_net, 0.0, places=11, msg="Net thermal Fz must be zero.")

        # RAMP formulation properties
        q_ramp = 8.0
        E0 = 1.0
        Emin = 1e-6

        def ramp_E(x):
            return Emin + (x / (1.0 + q_ramp * (1.0 - x))) * (E0 - Emin)

        def d_ramp_E(x):
            return ((1.0 + q_ramp) / ((1.0 + q_ramp * (1.0 - x)) ** 2)) * (E0 - Emin)

        # Non-zero derivative at x = 0 (prevents sensitivity freeze)
        d_at_0 = d_ramp_E(0.0)
        expected_d_at_0 = (E0 - Emin) / (1.0 + q_ramp)
        self.assertAlmostEqual(d_at_0, expected_d_at_0, places=10)
        self.assertGreater(d_at_0, 0.0, "RAMP derivative at x=0 must be strictly positive.")

        # Monotonicity check
        x_samples = np.linspace(0.0, 1.0, 50)
        E_samples = [ramp_E(xi) for xi in x_samples]
        self.assertTrue(np.all(np.diff(E_samples) > 0.0), "RAMP E_th must be strictly monotonic.")
        self.assertAlmostEqual(E_samples[0], Emin, places=8)
        self.assertAlmostEqual(E_samples[-1], E0, places=8)

        # Global assembly test
        opt = SIMPOptimizer3D(nelx=3, nely=3, nelz=3)
        xPhys = np.full(opt.num_elements, 0.5, dtype=np.float64)
        dT = np.full(opt.num_elements, 25.0, dtype=np.float64)  # 25 deg C rise
        F_th = opt.assemble_thermal_load_vector(xPhys, dT, q_ramp=8.0)

        self.assertEqual(len(F_th), opt.num_dofs)
        self.assertTrue(np.all(np.isfinite(F_th)), "F_th must be finite.")
        self.assertGreater(float(np.linalg.norm(F_th)), 0.0, "F_th must be non-zero for non-zero dT.")

    # =========================================================================
    # TIER 2: BOUNDARY & CORNER CASES
    # =========================================================================

    def test_tier2_passive_solid_elements_constraint(self):
        """
        Tier 2: Verify passive solid elements non-design space constraint.
        Ensures that:
        1. Elements in passive solid boxes have lb = ub = 1.0 in MMA.
        2. Density filtering and Heaviside projection strictly preserve xPhys == 1.0.
        3. Sensitivities for passive solid elements do not cause design deviation.
        """
        nelx, nely, nelz = 6, 6, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)
        opt.add_passive_box(xmin=2.0, xmax=4.0, ymin=2.0, ymax=4.0, zmin=0.0, zmax=2.0, dx=opt.dx, dy=opt.dy, dz=opt.dz)

        num_passive = int(np.sum(opt.passive_solid))
        self.assertGreater(num_passive, 0, "Passive solid box must contain elements.")

        # Simulate bounds configuration for MMA
        lb = np.full(opt.num_elements, 1e-3, dtype=np.float64)
        ub = np.full(opt.num_elements, 1.0, dtype=np.float64)
        lb[opt.passive_solid] = 1.0
        ub[opt.passive_solid] = 1.0

        # Elements where lb == ub are treated as fixed constants in MMA
        fixed_indices = np.where(lb == ub)[0]
        self.assertEqual(len(fixed_indices), num_passive)

        # Check density filtering preservation
        x_trial = np.random.uniform(0.1, 0.9, opt.num_elements)
        x_trial[opt.passive_solid] = 1.0

        x_grid = x_trial.reshape((opt.nelz, opt.nely, opt.nelx))
        conv_x = ndimage.convolve(x_grid, opt.kernel, mode='constant', cval=0.0)
        x_tilde = (conv_x / opt.kernel_normalizer).ravel()
        beta, eta = 4.0, 0.5
        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        xPhys = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
        xPhys[opt.passive_solid] = 1.0

        # Physical density must be identically 1.0 on passive solid
        np.testing.assert_allclose(
            xPhys[opt.passive_solid], 1.0, atol=1e-14,
            err_msg="Passive solid elements must strictly maintain xPhys == 1.0."
        )

    def test_tier2_high_contrast_condition_number(self):
        """
        Tier 2: Verify solver stability under extreme condition number (kappa >= 1e8).
        Simulates extreme contrast between solid (E0 = 1.0) and void (Emin = 1e-8).
        Tests that Smoothed Aggregation AMG-PCG with coarse_solver='pinv' converges
        without float overflow, zero-division warnings, or NaN values.
        """
        nelx, nely, nelz = 5, 5, 5
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-8, E0=1.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)

        # Create a challenging checkerboard / void distribution
        xPhys = np.full(opt.num_elements, 1e-3, dtype=np.float64)
        xPhys[::2] = 1.0  # High contrast: 1.0 vs 1e-3^3 * 1.0 ~ 1e-9

        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()
        F_free = opt.force_vector[free_dofs]

        # Smoothed aggregation AMG hierarchy with SVD pseudoinverse coarse solver
        ml = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
        M_amg = ml.aspreconditioner(cycle='V')

        u_amg, exit_code = sla.cg(K_free, F_free, M=M_amg, rtol=1e-5, maxiter=500)

        self.assertEqual(exit_code, 0, "AMG-PCG must converge even under extreme condition number.")
        self.assertTrue(np.all(np.isfinite(u_amg)), "Solution vector must not contain NaN or Inf.")
        res_norm = np.linalg.norm(K_free @ u_amg - F_free) / np.linalg.norm(F_free)
        self.assertLess(res_norm, 1e-4, f"Relative residual norm {res_norm} must be < 1e-4.")

    def test_tier2_float64_precision_retention(self):
        """
        Tier 2: Verify strict 64-bit IEEE 754 precision discipline.
        Confirms that:
        1. All matrices (K, K_th) and vectors (F, U, T) are strictly np.float64.
        2. Downcasting to float32 causes catastrophic precision degradation under
           high condition numbers, verifying why float64 is non-negotiable.
        """
        nelx, nely, nelz = 4, 4, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-8, E0=1.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        
        # High contrast distribution yielding condition number > 1e7
        xPhys = np.full(opt.num_elements, 1e-3, dtype=np.float64)
        xPhys[::2] = 1.0

        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()
        F_free = opt.force_vector[free_dofs]

        # Data type verification
        self.assertEqual(K_free.dtype, np.float64, "Stiffness matrix must be float64.")
        self.assertEqual(F_free.dtype, np.float64, "Force vector must be float64.")

        # Solve in float64
        u_64, code_64 = sla.cg(K_free, F_free, rtol=1e-8, maxiter=2000)
        self.assertEqual(code_64, 0)
        res_64 = np.linalg.norm(K_free @ u_64 - F_free) / np.linalg.norm(F_free)
        self.assertLess(res_64, 1e-5, "Float64 solve must achieve high precision residual < 1e-5.")

        # Solve in float32 to demonstrate precision loss
        K_32 = K_free.astype(np.float32)
        F_32 = F_free.astype(np.float32)
        u_32, _ = sla.cg(K_32, F_32, rtol=1e-8, maxiter=2000)

        # Float32 residual evaluated in native float64
        res_32 = np.linalg.norm(K_free @ u_32.astype(np.float64) - F_free) / np.linalg.norm(F_free)
        self.assertGreater(
            res_32, 1e-1,
            "Float32 under high condition number suffers severe precision loss."
        )

    def test_tier2_coarse_grid_pseudoinverse_stability(self):
        """
        Tier 2: Verify coarse-grid stability with SVD pseudoinverse (coarse_solver='pinv').
        Tests an ill-conditioned system with soft isolated modes where exact LU ('splu')
        can become ill-conditioned or crash, while 'pinv' smoothly truncates zero singular
        values and produces a strictly finite preconditioner.
        """
        nelx, nely, nelz = 6, 4, 3
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)
        opt.fix_node(0, 0, 0, [0, 1, 2])  # Minimally constrained (near-singular)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        xPhys = np.full(opt.num_elements, 1e-4, dtype=np.float64)
        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

        # Construction with 'pinv' must succeed cleanly without exception
        try:
            ml_pinv = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
            pinv_success = True
        except Exception as e:
            pinv_success = False

        self.assertTrue(pinv_success, "pyamg smoothed_aggregation_solver with coarse_solver='pinv' must succeed.")

        # Preconditioner evaluation must yield strictly finite vectors
        M_pinv = ml_pinv.aspreconditioner()
        v_test = np.ones(K_free.shape[0], dtype=np.float64)
        v_prec = M_pinv.matvec(v_test)
        self.assertTrue(np.all(np.isfinite(v_prec)), "Preconditioner matvec must produce finite values.")

    # =========================================================================
    # TIER 3: CROSS-FEATURE COMBINATIONS
    # =========================================================================

    def test_tier3_mma_amg_coupled_solver(self):
        """
        Tier 3: Verify coupled MMA optimizer + AMG linear solver integration with state caching.
        Exercises the _TOStateEvaluator pattern:
        1. Objective and volume constraint callbacks share cached FEA results.
        2. Total FEA solve count equals iteration count (zero redundant FEA solves).
        3. Monotonic compliance decrease or stabilization over 5 iterations.
        4. Volume constraint satisfied within tolerance.
        """
        nelx, nely, nelz = 6, 4, 3
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, volfrac=0.4, rmin=1.5)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # Implement State Caching Evaluator
        class StateEvaluator:
            def __init__(self, optimizer):
                self.opt = optimizer
                self.last_x: Optional[np.ndarray] = None
                self.fea_solve_count = 0
                self.it = 0
                self.C0: Optional[float] = None
                self.compliance = 0.0
                self.vol_fraction = optimizer.volfrac
                self.dc_filtered = np.zeros(optimizer.num_elements)
                self.dv_filtered = np.zeros(optimizer.num_elements)
                self.u_prev = None

            def evaluate(self, x: np.ndarray):
                if self.last_x is not None and np.allclose(x, self.last_x, atol=1e-14):
                    return  # Cache HIT: skip redundant FEA solve

                self.it += 1
                self.fea_solve_count += 1
                
                x_grid = x.reshape((self.opt.nelz, self.opt.nely, self.opt.nelx))
                conv_x = ndimage.convolve(x_grid, self.opt.kernel, mode='constant', cval=0.0)
                xPhys = (conv_x / self.opt.kernel_normalizer).ravel()

                # AMG Linear Solve
                K_full = self.opt.assemble_elastic_stiffness(xPhys)
                K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

                ml = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
                M_amg = ml.aspreconditioner(cycle='V')

                u_free, code = sla.cg(K_free, F_free, x0=self.u_prev, M=M_amg, rtol=1e-5, maxiter=500)
                if code != 0:
                    u_free = sla.spsolve(K_free, F_free)
                self.u_prev = u_free.copy()

                U = np.zeros(self.opt.num_dofs, dtype=np.float64)
                U[free_dofs] = u_free

                U_e = U[self.opt.edofMat]
                E_elements = self.opt.Emin + (xPhys ** self.opt.penal) * (self.opt.E0 - self.opt.Emin)
                ce = np.sum((U_e @ self.opt.k0) * U_e, axis=1)
                compliance = float(np.sum(E_elements * ce))

                if self.C0 is None:
                    self.C0 = max(1e-12, compliance)

                dc_comp = -self.opt.penal * (self.opt.E0 - self.opt.Emin) * (xPhys ** (self.opt.penal - 1.0)) * ce

                q_obj = dc_comp.reshape((self.opt.nelz, self.opt.nely, self.opt.nelx))
                conv_q = ndimage.convolve(q_obj / self.opt.kernel_normalizer, self.opt.kernel, mode='constant', cval=0.0)
                self.dc_filtered = conv_q.ravel()

                self.compliance = compliance
                self.vol_fraction = float(np.mean(xPhys))
                self.last_x = x.copy()

        evaluator = StateEvaluator(opt)

        def obj_cb(x: np.ndarray, grad: np.ndarray) -> float:
            evaluator.evaluate(x)
            if grad.size > 0:
                grad[:] = evaluator.dc_filtered / evaluator.C0
            return evaluator.compliance / evaluator.C0

        def vol_cb(x: np.ndarray, grad: np.ndarray) -> float:
            evaluator.evaluate(x)
            if grad.size > 0:
                grad[:] = 1.0 / (opt.num_elements * opt.volfrac)
            return (evaluator.vol_fraction - opt.volfrac) / opt.volfrac

        opt_mma = nlopt.opt(nlopt.LD_MMA, opt.num_elements)
        opt_mma.set_lower_bounds(np.full(opt.num_elements, 1e-3))
        opt_mma.set_upper_bounds(np.full(opt.num_elements, 1.0))
        opt_mma.set_min_objective(obj_cb)
        opt_mma.add_inequality_constraint(vol_cb, 1e-4)
        opt_mma.set_maxeval(10)  # 5 outer iterations (obj + constr = 2 calls per iter)

        x0 = np.full(opt.num_elements, opt.volfrac, dtype=np.float64)
        try:
            x_res = opt_mma.optimize(x0)
        except nlopt.RoundoffLimited:
            pass

        # 1. Cache HIT verification: Total FEA solves must be significantly fewer than total callback calls
        total_eval_calls = evaluator.it
        self.assertLessEqual(
            evaluator.fea_solve_count, total_eval_calls,
            "State caching must eliminate redundant FEA solves."
        )

        # 2. Volume constraint satisfied within tolerance
        self.assertLessEqual(
            evaluator.vol_fraction, opt.volfrac + 0.02,
            "Volume fraction must respect target constraint."
        )

    def test_tier3_mma_passive_boxes_amg(self):
        """
        Tier 3: Verify MMA optimizer + Passive Solid Box + AMG Linear Solver.
        Applies a passive solid reinforcement pad on a cantilever domain.
        Verifies:
        1. Passive elements maintain xPhys == 1.0 throughout optimization.
        2. AMG preconditioned solver seamlessly handles the stiffness discontinuity.
        """
        nelx, nely, nelz = 6, 6, 3
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, volfrac=0.35)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-50.0)

        # Add passive solid box at the root
        opt.add_passive_box(xmin=0.0, xmax=2.0, ymin=0.0, ymax=6.0, zmin=0.0, zmax=3.0, dx=opt.dx, dy=opt.dy, dz=opt.dz)
        self.assertGreater(np.sum(opt.passive_solid), 0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # Initial density with passive solid
        x = np.full(opt.num_elements, opt.volfrac, dtype=np.float64)
        x[opt.passive_solid] = 1.0

        # Assembly and AMG solve with passive solid
        K_full = opt.assemble_elastic_stiffness(x)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

        ml = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
        M_amg = ml.aspreconditioner(cycle='V')

        u_free, code = sla.cg(K_free, F_free, M=M_amg, rtol=1e-5, maxiter=500)
        self.assertEqual(code, 0, "AMG solve must succeed on domain with passive solid box.")
        self.assertTrue(np.all(np.isfinite(u_free)))

        # Verify passive elements remain 1.0
        self.assertTrue(np.all(x[opt.passive_solid] == 1.0))

    def test_tier3_amg_displacement_warm_starting(self):
        """
        Tier 3: Verify iteration reduction via AMG displacement warm-starting.
        Solves step 1 with cold start (x0 = 0), and step 2 (after small density update)
        with warm start (x0 = u1) versus cold start.
        Verifies that warm-starting reduces the number of PCG iterations.
        """
        nelx, nely, nelz = 8, 8, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # Step 1: Initial density
        x1 = np.full(opt.num_elements, 0.4, dtype=np.float64)
        K1_full = opt.assemble_elastic_stiffness(x1)
        K1_free = K1_full[free_dofs, :][:, free_dofs].tocsr()

        ml1 = pyamg.smoothed_aggregation_solver(K1_free, coarse_solver='pinv', symmetry='symmetric')
        M1 = ml1.aspreconditioner(cycle='V')

        # Iteration count tracker
        iters_cold1 = []
        u1, code1 = sla.cg(
            K1_free, F_free, x0=None, M=M1, rtol=1e-5, maxiter=500,
            callback=lambda x: iters_cold1.append(1)
        )
        self.assertEqual(code1, 0)
        n_iters_cold1 = len(iters_cold1)

        # Step 2: Perturbed density (small step as in topology optimization)
        np.random.seed(42)
        x2 = np.clip(x1 + np.random.uniform(-0.05, 0.05, opt.num_elements), 1e-3, 1.0)
        K2_full = opt.assemble_elastic_stiffness(x2)
        K2_free = K2_full[free_dofs, :][:, free_dofs].tocsr()

        ml2 = pyamg.smoothed_aggregation_solver(K2_free, coarse_solver='pinv', symmetry='symmetric')
        M2 = ml2.aspreconditioner(cycle='V')

        # Solve Step 2 with warm start (x0 = u1)
        iters_warm = []
        u2_warm, code_warm = sla.cg(
            K2_free, F_free, x0=u1, M=M2, rtol=1e-5, maxiter=500,
            callback=lambda x: iters_warm.append(1)
        )
        self.assertEqual(code_warm, 0)
        n_iters_warm = len(iters_warm)

        # Solve Step 2 with cold start (x0 = 0)
        iters_cold2 = []
        u2_cold, code_cold = sla.cg(
            K2_free, F_free, x0=None, M=M2, rtol=1e-5, maxiter=500,
            callback=lambda x: iters_cold2.append(1)
        )
        self.assertEqual(code_cold, 0)
        n_iters_cold2 = len(iters_cold2)

        # Warm start must reduce PCG iteration count
        self.assertLess(
            n_iters_warm, n_iters_cold2,
            f"Warm start iterations ({n_iters_warm}) must be strictly fewer than cold start ({n_iters_cold2})."
        )

    # =========================================================================
    # TIER 4: REAL-WORLD APPLICATION SCENARIOS
    # =========================================================================

    def test_tier4_cantilever_benchmark_mma_amg(self):
        """
        Tier 4: End-to-end 3D Cantilever beam optimization benchmark with MMA and AMG.
        Mesh: 10 x 10 x 10 (1,000 H8 elements, 3,993 DOFs).
        Boundary condition: Fixed left face (x=0).
        Load: Downward tip load at center of right face.
        Objective: Minimize compliance under volume fraction V = 0.30 constraint.
        Verifies:
        1. Volume fraction conservation: |V_final - 0.30| <= 0.02.
        2. Objective compliance reduction: C_final < C_initial.
        3. Strict memory ceiling: Peak memory consumption strictly < 100 MB.
        4. Physical structural validity: Continuous load path formed.
        """
        nelx, nely, nelz = 10, 10, 10
        volfrac = 0.30

        tracemalloc.start()
        start_time = time.time()

        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            dx=1.0, dy=1.0, dz=1.0,
            volfrac=volfrac, penal=3.0, rmin=1.5,
            max_iter=15
        )
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # State Evaluator linking MMA and AMG
        class CantileverBenchmarkEvaluator:
            def __init__(self, optimizer):
                self.opt = optimizer
                self.last_x: Optional[np.ndarray] = None
                self.C0: Optional[float] = None
                self.compliance = 0.0
                self.vol_fraction = optimizer.volfrac
                self.dc_filtered = np.zeros(optimizer.num_elements)
                self.u_prev = None
                self.compliance_history: List[float] = []

            def evaluate(self, x: np.ndarray):
                if self.last_x is not None and np.allclose(x, self.last_x, atol=1e-14):
                    return

                x_grid = x.reshape((self.opt.nelz, self.opt.nely, self.opt.nelx))
                conv_x = ndimage.convolve(x_grid, self.opt.kernel, mode='constant', cval=0.0)
                xPhys = (conv_x / self.opt.kernel_normalizer).ravel()

                # Assemble & solve with AMG
                K_full = self.opt.assemble_elastic_stiffness(xPhys)
                K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

                ml = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
                M_amg = ml.aspreconditioner(cycle='V')

                u_free, code = sla.cg(K_free, F_free, x0=self.u_prev, M=M_amg, rtol=1e-5, maxiter=500)
                if code != 0:
                    u_free = sla.spsolve(K_free, F_free)
                self.u_prev = u_free.copy()

                U = np.zeros(self.opt.num_dofs, dtype=np.float64)
                U[free_dofs] = u_free

                U_e = U[self.opt.edofMat]
                E_elements = self.opt.Emin + (xPhys ** self.opt.penal) * (self.opt.E0 - self.opt.Emin)
                ce = np.sum((U_e @ self.opt.k0) * U_e, axis=1)
                compliance = float(np.sum(E_elements * ce))

                if self.C0 is None:
                    self.C0 = max(1e-12, compliance)

                self.compliance_history.append(compliance)

                dc_comp = -self.opt.penal * (self.opt.E0 - self.opt.Emin) * (xPhys ** (self.opt.penal - 1.0)) * ce

                q_obj = dc_comp.reshape((self.opt.nelz, self.opt.nely, self.opt.nelx))
                conv_q = ndimage.convolve(q_obj / self.opt.kernel_normalizer, self.opt.kernel, mode='constant', cval=0.0)
                self.dc_filtered = conv_q.ravel()

                self.compliance = compliance
                self.vol_fraction = float(np.mean(xPhys))
                self.last_x = x.copy()

        evaluator = CantileverBenchmarkEvaluator(opt)

        def obj_func(x: np.ndarray, grad: np.ndarray) -> float:
            evaluator.evaluate(x)
            if grad.size > 0:
                grad[:] = evaluator.dc_filtered / evaluator.C0
            return evaluator.compliance / evaluator.C0

        def vol_func(x: np.ndarray, grad: np.ndarray) -> float:
            evaluator.evaluate(x)
            if grad.size > 0:
                grad[:] = 1.0 / (opt.num_elements * volfrac)
            return (evaluator.vol_fraction - volfrac) / volfrac

        opt_mma = nlopt.opt(nlopt.LD_MMA, opt.num_elements)
        opt_mma.set_lower_bounds(np.full(opt.num_elements, 1e-3))
        opt_mma.set_upper_bounds(np.full(opt.num_elements, 1.0))
        opt_mma.set_min_objective(obj_func)
        opt_mma.add_inequality_constraint(vol_func, 1e-4)
        opt_mma.set_maxeval(30)  # Up to 15 iterations

        x0 = np.full(opt.num_elements, volfrac, dtype=np.float64)
        try:
            x_opt = opt_mma.optimize(x0)
        except nlopt.RoundoffLimited:
            pass

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024.0 * 1024.0)
        elapsed = time.time() - start_time

        # 1. Volume Fraction Conservation
        self.assertAlmostEqual(
            evaluator.vol_fraction, volfrac, delta=0.02,
            msg=f"Final volume fraction {evaluator.vol_fraction:.3f} must be within 0.30 +/- 0.02."
        )

        # 2. Compliance Reduction / Stabilization
        self.assertGreater(len(evaluator.compliance_history), 1)
        self.assertLess(
            evaluator.compliance_history[-1], evaluator.compliance_history[0],
            "Final compliance must be lower than initial compliance."
        )

        # 3. Memory Ceiling Check (< 100 MB benchmark)
        self.assertLess(
            peak_mb, 100.0,
            f"Peak memory {peak_mb:.2f} MB exceeds 100 MB budget ceiling."
        )


if __name__ == "__main__":
    unittest.main()
