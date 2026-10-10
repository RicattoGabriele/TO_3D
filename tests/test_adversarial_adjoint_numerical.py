"""
Adversarial Stress Test Suite: Adjoint Sensitivities & Numerical Solvers for CE-3D.
Author: challenger_adjoint_numerical (Milestone M5 Empirical Verifier)

Scope:
1. Analytical vs Central Finite Difference Sensitivities for dC_th/dx_e:
   - Probes across random densities x in (0.01, 0.99), multiple grid dimensions, and diverse thermal loads.
   - Element-wise relative error verification: max_e |dC_th/dx_e - FD| / |FD| <= 1e-6.
   - Strict physical sign preservation: dC_th/dx_e <= 0 everywhere, zero NaNs/Infs.
2. MMA Optimization Convergence under Thermal Compliance Mode:
   - Verifies monotonic decrease of thermal compliance C_th without NaN or stall across iterations.
   - Checks satisfaction of volume fraction constraints (|V_final - V_target| <= 0.015).
3. AMG and PCG Linear Solver Convergence Accuracy under High Density Contrasts (1e-9 to 1.0):
   - Constructs Poisson conduction systems with extreme conductivity contrasts (up to 1e9).
   - Validates relative residual norm < 1e-5 and relative solution error vs direct SuperLU < 1e-5.
"""

import os
import sys
import unittest
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_3d import (
    SIMPOptimizer3D,
    SIMPResult3D,
    solve_linear_system
)


class TestAdjointSensitivitiesAdversarial(unittest.TestCase):
    """Adversarial validation of thermal compliance adjoint sensitivities against finite differences."""

    def test_analytical_vs_central_fd_multiple_grids_and_loads(self):
        """
        Stress Test 1A: Analytical sensitivities vs central finite differences (h = 1e-6)
        across multiple grid dimensions and diverse thermal load / sink configurations.
        Evaluates random densities x in [0.05, 0.95].
        Verifies:
        1. max_e |dC_th/dx_e - FD| / |FD| <= 1.0e-6 across all elements.
        2. dC_th/dx_e <= 0 everywhere (adding conduction never increases thermal compliance).
        3. Zero NaNs and zero Infs.
        """
        test_configs = [
            # (nelx, nely, nelz, bc_name, seed)
            (4, 4, 2, "single_face_sink_corner_source", 101),
            (3, 3, 3, "top_face_sink_center_source", 202),
            (6, 3, 2, "opposing_face_sinks_center_source", 303),
            (3, 4, 3, "bottom_face_sink_top_distributed_source", 404),
        ]

        h = 1.0e-6
        penal_th = 3.0

        for nelx, nely, nelz, bc_name, seed in test_configs:
            with self.subTest(grid=f"{nelx}x{nely}x{nelz}", bc=bc_name):
                opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")

                if bc_name == "single_face_sink_corner_source":
                    opt.fix_thermal_face("left", temp=0.0)
                    opt.add_heat_source(nelx, nely, nelz, q=1.0)
                elif bc_name == "top_face_sink_center_source":
                    opt.fix_thermal_face("top", temp=0.0)
                    opt.add_heat_source(nelx // 2, nely // 2, 0, q=1.0)
                elif bc_name == "opposing_face_sinks_center_source":
                    opt.fix_thermal_face("left", temp=0.0)
                    opt.fix_thermal_face("right", temp=0.0)
                    opt.add_heat_source(nelx // 2, nely // 2, nelz // 2, q=1.0)
                elif bc_name == "bottom_face_sink_top_distributed_source":
                    opt.fix_thermal_face("bottom", temp=0.0)
                    for ix in range(nelx + 1):
                        for iz in range(nelz + 1):
                            opt.add_heat_source(ix, nely, iz, q=0.2)

                # Randomized densities in [0.05, 0.95]
                rng = np.random.RandomState(seed)
                x_base = rng.uniform(0.05, 0.95, size=opt.num_elements)

                T_base = opt.solve_thermal(x_base, penal_th=penal_th, solver_type="direct")
                C_base, dc_ana = opt.compute_thermal_compliance(x_base, T_base, penal_th=penal_th)
                free_th = [n for n in range(opt.num_nodes) if n not in opt.fixed_thermal_nodes]

                # Sign check
                self.assertLessEqual(
                    float(np.max(dc_ana)), 1e-12,
                    f"Sensitivities must be non-positive on grid {nelx}x{nely}x{nelz}."
                )
                self.assertFalse(np.any(np.isnan(dc_ana)), "Sensitivities contain NaNs.")
                self.assertFalse(np.any(np.isinf(dc_ana)), "Sensitivities contain Infs.")

                # Central finite differences
                max_rel_err = 0.0
                worst_elem = -1

                for e in range(opt.num_elements):
                    x_p = x_base.copy()
                    x_p[e] += h
                    T_p = opt.solve_thermal(x_p, penal_th=penal_th, solver_type="direct")
                    C_p = float(np.dot(opt.heat_source_vector[free_th], T_p[free_th]))

                    x_m = x_base.copy()
                    x_m[e] -= h
                    T_m = opt.solve_thermal(x_m, penal_th=penal_th, solver_type="direct")
                    C_m = float(np.dot(opt.heat_source_vector[free_th], T_m[free_th]))

                    fd_e = (C_p - C_m) / (2.0 * h)
                    ana_e = dc_ana[e]

                    if abs(fd_e) > 1e-9:
                        rel_err = abs(ana_e - fd_e) / abs(fd_e)
                        if rel_err > max_rel_err:
                            max_rel_err = rel_err
                            worst_elem = e

                print(f"[Adjoint FD 1A] Grid {nelx}x{nely}x{nelz} | BC: {bc_name} | Max Rel Err: {max_rel_err:.3e} (elem {worst_elem})")
                self.assertLessEqual(
                    max_rel_err, 1.0e-6,
                    f"Grid {nelx}x{nely}x{nelz} max relative error {max_rel_err:.3e} exceeds 1.0e-6."
                )

    def test_analytical_vs_fd_full_density_spectrum_001_to_099(self):
        """
        Stress Test 1B: Analytical sensitivities probed across the full density spectrum
        x in (0.01, 0.99) spanning void-like elements (x=0.01, 0.02, 0.05), intermediate (x=0.3, 0.5),
        and solid-like elements (x=0.8, 0.99).
        Verifies:
        1. max_e |dC_th/dx_e - FD| / |FD| <= 1.0e-6 across the entire range with step h = 2e-5.
        2. All sensitivities are strictly non-positive (dC_th/dx_e <= 0).
        """
        nelx, nely, nelz = 5, 4, 3
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")
        opt.fix_thermal_face("left", temp=0.0)
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=1.0)

        # Diverse density spectrum including boundary values
        rng = np.random.RandomState(999)
        x_base = rng.uniform(0.01, 0.99, size=opt.num_elements)
        # Ensure explicit extreme elements exist
        x_base[0] = 0.011
        x_base[1] = 0.025
        x_base[2] = 0.985
        x_base[3] = 0.990

        T_base = opt.solve_thermal(x_base, solver_type="direct")
        C_base, dc_ana = opt.compute_thermal_compliance(x_base, T_base)
        free_th = [n for n in range(opt.num_nodes) if n not in opt.fixed_thermal_nodes]

        h = 2.0e-5  # Minimizes float64 subtractive cancellation on elements with x near 0.01
        max_rel_err = 0.0
        worst_elem = -1

        for e in range(opt.num_elements):
            x_p = x_base.copy()
            x_p[e] += h
            T_p = opt.solve_thermal(x_p, solver_type="direct")
            C_p = float(np.dot(opt.heat_source_vector[free_th], T_p[free_th]))

            x_m = x_base.copy()
            x_m[e] -= h
            T_m = opt.solve_thermal(x_m, solver_type="direct")
            C_m = float(np.dot(opt.heat_source_vector[free_th], T_m[free_th]))

            fd_e = (C_p - C_m) / (2.0 * h)
            ana_e = dc_ana[e]

            if abs(fd_e) > 1e-9:
                rel_err = abs(ana_e - fd_e) / abs(fd_e)
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                    worst_elem = e

        print(f"[Adjoint FD 1B] Full Spectrum (0.01..0.99) Grid {nelx}x{nely}x{nelz} | Max Rel Err: {max_rel_err:.3e} (elem {worst_elem}, x={x_base[worst_elem]:.3f})")
        self.assertLessEqual(
            max_rel_err, 1.0e-6,
            f"Full spectrum (0.01 to 0.99) max relative error {max_rel_err:.3e} exceeds 1.0e-6."
        )


class TestMMAOptimizationConvergenceThermalCompliance(unittest.TestCase):
    """Adversarial validation of MMA optimizer convergence in thermal compliance mode."""

    def test_mma_thermal_compliance_monotonic_descent_standard_sink(self):
        """
        Stress Test 2A: Verifies that MMA optimization on standard 3D heat sink
        achieves strict monotonic decrease of thermal compliance C_th without NaN or stall.
        """
        nelx, nely, nelz = 8, 6, 4
        volfrac = 0.35
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, rmin=1.5,
            solver_type="direct", optimizer_type="mma"
        )
        opt.fix_thermal_face("left", temp=0.0)
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=25.0)

        res = opt.solve(max_iter=20, tol=0.005, mode="thermal_compliance", optimizer_type="mma")

        self.assertTrue(res.success, "Optimizer must succeed.")
        self.assertGreaterEqual(res.iterations_run, 15, "Must perform meaningful optimization iterations.")

        c_hist = res.compliance_history
        self.assertFalse(np.any(np.isnan(c_hist)), "Compliance history must not contain NaNs.")
        self.assertFalse(np.any(np.isinf(c_hist)), "Compliance history must not contain Infs.")

        # Strict monotonic decrease check
        diffs = np.diff(c_hist)
        num_increases = int(np.sum(diffs > 1e-9))
        self.assertEqual(
            num_increases, 0,
            f"Thermal compliance must decrease strictly monotonically, but found {num_increases} increases."
        )

        c_reduction = (c_hist[0] - c_hist[-1]) / c_hist[0]
        print(f"[MMA Conv 2A] Standard Sink: {res.iterations_run} iters | C_init={c_hist[0]:.2e} -> C_final={c_hist[-1]:.2e} (-{c_reduction*100:.1f}%) | Monotonic: True | Vf={res.volume_fraction:.4f}")
        self.assertGreater(c_reduction, 0.80, f"Expected >80% compliance reduction, got {c_reduction*100:.1f}%.")

        # Volume fraction satisfaction
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.015,
                               msg=f"Volume fraction {res.volume_fraction:.4f} did not match target {volfrac}.")

    def test_mma_thermal_compliance_multisource_corner_sinks(self):
        """
        Stress Test 2B: Verifies MMA convergence under complex thermal topology:
        dual cold boundary walls and central concentrated heat source.
        """
        nelx, nely, nelz = 6, 4, 4
        volfrac = 0.40
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, rmin=1.5,
            solver_type="direct", optimizer_type="mma"
        )
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=0.0)
        opt.add_heat_source(nelx // 2, nely // 2, nelz // 2, q=40.0)

        res = opt.solve(max_iter=18, tol=0.005, mode="thermal_compliance", optimizer_type="mma")

        self.assertTrue(res.success)
        c_hist = res.compliance_history
        diffs = np.diff(c_hist)
        num_increases = int(np.sum(diffs > 1e-9))
        self.assertEqual(
            num_increases, 0,
            f"Thermal compliance must decrease strictly monotonically, but found {num_increases} increases."
        )

        c_reduction = (c_hist[0] - c_hist[-1]) / c_hist[0]
        print(f"[MMA Conv 2B] Multisource: {res.iterations_run} iters | C_init={c_hist[0]:.2e} -> C_final={c_hist[-1]:.2e} (-{c_reduction*100:.1f}%) | Monotonic: True | Vf={res.volume_fraction:.4f}")
        self.assertGreater(c_reduction, 0.80, f"Expected >80% compliance reduction, got {c_reduction*100:.1f}%.")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.015)


class TestAMGandPCGSolversHighDensityContrast(unittest.TestCase):
    """Adversarial stress tests for AMG and PCG solvers under extreme density contrasts."""

    def test_amg_and_pcg_extreme_contrast_1e9(self):
        """
        Stress Test 3A: Solves the thermal conductivity Poisson equation with an extreme
        conductivity contrast of 1e9 (k_min = 1e-9 up to k = 1.0).
        Verifies:
        1. SA-AMG relative residual norm < 1.0e-5 and relative solution error vs SuperLU < 1.0e-5.
        2. Jacobi-PCG relative residual norm < 1.0e-5 and relative solution error vs SuperLU < 1.0e-5.
        3. Zero NaNs or Infs in solution fields.
        """
        nelx, nely, nelz = 8, 6, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-9, solver_type="direct")
        opt.fix_thermal_face("left", temp=0.0)
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=50.0)

        # 50% void (x = 1e-9), 50% solid (x = 1.0)
        rng = np.random.RandomState(42)
        xPhys = np.where(rng.rand(opt.num_elements) > 0.5, 1.0, 1e-9)

        K_th = opt.assemble_thermal_conductivity(xPhys, penal_th=3.0)
        fixed_th_set = set(opt.fixed_thermal_nodes.keys())
        free_th = np.array([n for n in range(opt.num_nodes) if n not in fixed_th_set], dtype=np.int32)

        K_free = K_th[free_th, :][:, free_th]
        Q_free = opt.heat_source_vector[free_th].copy()

        # Direct SuperLU benchmark
        T_direct, _, _ = solve_linear_system(K_free, Q_free, solver_type="direct")
        norm_direct = np.linalg.norm(T_direct)
        norm_Q = np.linalg.norm(Q_free)

        # SA-AMG Solver
        T_amg, code_amg, solver_amg = solve_linear_system(K_free, Q_free, solver_type="amg", rtol=1e-6, maxiter=2000)
        self.assertEqual(code_amg, 0, "AMG solver must converge with exit code 0.")
        self.assertEqual(solver_amg, "amg_pcg", "Must successfully use amg_pcg solver without falling back.")
        res_amg = np.linalg.norm(K_free @ T_amg - Q_free) / norm_Q
        err_amg = np.linalg.norm(T_amg - T_direct) / norm_direct

        # Jacobi-PCG Solver
        T_pcg, code_pcg, solver_pcg = solve_linear_system(K_free, Q_free, solver_type="pcg", rtol=1e-6, maxiter=2000)
        self.assertEqual(code_pcg, 0, "PCG solver must converge with exit code 0.")
        self.assertEqual(solver_pcg, "jacobi_pcg", "Must successfully use jacobi_pcg solver.")
        res_pcg = np.linalg.norm(K_free @ T_pcg - Q_free) / norm_Q
        err_pcg = np.linalg.norm(T_pcg - T_direct) / norm_direct

        print(f"[Solvers 3A] Contrast 1e9 | AMG res={res_amg:.2e}, err={err_amg:.2e} | PCG res={res_pcg:.2e}, err={err_pcg:.2e}")
        self.assertLess(res_amg, 1.0e-5, f"AMG relative residual {res_amg:.2e} must be < 1e-5.")
        self.assertLess(err_amg, 1.0e-5, f"AMG relative solution error {err_amg:.2e} must be < 1e-5.")
        self.assertLess(res_pcg, 1.0e-5, f"PCG relative residual {res_pcg:.2e} must be < 1e-5.")
        self.assertLess(err_pcg, 1.0e-5, f"PCG relative solution error {err_pcg:.2e} must be < 1e-5.")

    def test_amg_and_pcg_contrast_sweep_1e4_to_1e9(self):
        """
        Stress Test 3B: Sweeps conductivity contrast from 1e4 to 1e9 across grids,
        verifying monotonic stability and that neither AMG nor PCG breakdown or diverge.
        """
        contrasts = [1e-4, 1e-6, 1e-8, 1e-9]
        nelx, nely, nelz = 6, 4, 4

        for k_min in contrasts:
            with self.subTest(k_min=k_min):
                opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=k_min, solver_type="direct")
                opt.fix_thermal_face("left", temp=0.0)
                opt.add_heat_source(nelx, nely // 2, nelz // 2, q=20.0)

                rng = np.random.RandomState(int(-np.log10(k_min)))
                xPhys = np.where(rng.rand(opt.num_elements) > 0.5, 1.0, 1e-3)

                K_th = opt.assemble_thermal_conductivity(xPhys, penal_th=3.0)
                fixed_th_set = set(opt.fixed_thermal_nodes.keys())
                free_th = np.array([n for n in range(opt.num_nodes) if n not in fixed_th_set], dtype=np.int32)

                K_free = K_th[free_th, :][:, free_th]
                Q_free = opt.heat_source_vector[free_th].copy()

                T_direct, _, _ = solve_linear_system(K_free, Q_free, solver_type="direct")
                norm_direct = np.linalg.norm(T_direct)

                T_amg, code_amg, _ = solve_linear_system(K_free, Q_free, solver_type="amg", rtol=1e-6)
                self.assertEqual(code_amg, 0)
                err_amg = np.linalg.norm(T_amg - T_direct) / norm_direct
                self.assertLess(err_amg, 1.0e-5)

                T_pcg, code_pcg, _ = solve_linear_system(K_free, Q_free, solver_type="pcg", rtol=1e-6)
                self.assertEqual(code_pcg, 0)
                err_pcg = np.linalg.norm(T_pcg - T_direct) / norm_direct
                self.assertLess(err_pcg, 1.0e-5)
                print(f"[Solvers 3B] Contrast 1e{int(-np.log10(k_min))} | AMG err={err_amg:.2e} | PCG err={err_pcg:.2e}")


if __name__ == "__main__":
    unittest.main()
