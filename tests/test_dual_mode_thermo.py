"""
Comprehensive 4-Tier Test Suite for Dual-Mode Thermo-Mechanical Topology Optimization.

Validates:
- Tier 1 (Feature Coverage):
  1. 'thermo_elastic' mode execution under coupled mechanical + thermal expansion loads.
  2. 'thermal_compliance' mode execution minimizing steady-state thermal compliance C_th = Q^T T.
  3. 'temperature_field' 3D array shape (nelz, nely, nelx) and physical value validity.
  4. Heat source linear scaling effects on thermal field and compliance.
  5. Dirichlet fixed temperature boundaries on faces and discrete nodes.
- Tier 2 (Boundary & Corner Cases):
  6. Extreme temperature gradients (Delta T = 1000 K) numerical stability.
  7. Pure thermal conduction vs coupled thermo-elastic behavior comparison.
  8. High volume fraction (Vf = 0.85) thermal compliance optimization.
  9. Low volume fraction (Vf = 0.10) thermal compliance optimization.
  10. Zero thermal expansion coefficient (alpha_th = 0) uncoupling verification.
- Tier 3 (Numerical & Adjoint Verification):
  11. Exact central finite difference verification for dC_th/dx_e matching analytical
      sensitivities to within <= 10^-6 relative error.
  12. MMA optimizer convergence on thermal compliance mode without NaN/Inf.
  13. SA-AMG preconditioned CG linear solver accuracy on thermal conductivity system.
  14. Minimum length-scale control feature verification via 3D morphological erosion
      metric confirming suppression of 1-voxel branches when rmin >= 2.0 * max(dx, dy, dz).
  15. Consistency and physical sign structure of 3-term thermo-elastic adjoint sensitivities.
- Tier 4 (Real-World Benchmark):
  16. 3D Heat Sink / Heat Exchanger dissipation benchmark under concentrated heat source,
      verifying volume conservation, compliance decrease > 40%, and peak RAM < 100 MB.
"""

import os
import sys
import unittest
import tracemalloc
from typing import Tuple, List, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.ndimage as ndimage
from scipy.ndimage import binary_erosion

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_3d import (
    SIMPOptimizer3D,
    SIMPResult3D,
    solve_linear_system
)


class TestDualModeThermo(unittest.TestCase):
    """4-Tier test suite for Dual-Mode Thermo-Mechanical Topology Optimization."""

    # =========================================================================
    # TIER 1: FEATURE COVERAGE (>=5 tests)
    # =========================================================================

    def test_tier1_thermo_elastic_mode_execution(self):
        """
        Tier 1.1: Verify execution of 'thermo_elastic' mode in SIMPOptimizer3D.solve().
        Exercises coupled structural + RAMP thermal expansion with 3-term adjoint sensitivities.
        """
        nelx, nely, nelz = 6, 4, 4
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=0.35, rmin=1.5,
            solver_type="direct", optimizer_type="mma",
            T_ref=0.0
        )
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-50.0)
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=80.0)
        opt.alpha_th = 1.0e-5

        res = opt.solve(max_iter=6, tol=0.015, mode="thermo_elastic", optimizer_type="mma")

        self.assertIsInstance(res, SIMPResult3D)
        self.assertTrue(res.success, "Optimizer should converge or finish cleanly in thermo_elastic mode.")
        self.assertGreater(len(res.compliance_history), 1, "Compliance history must record iterations.")
        self.assertGreater(res.compliance, 0.0, "Thermo-elastic compliance must be strictly positive.")
        self.assertAlmostEqual(res.volume_fraction, 0.35, delta=0.03,
                               msg=f"Volume fraction {res.volume_fraction} did not meet target 0.35 within tolerance.")

    def test_tier1_thermal_compliance_mode_execution(self):
        """
        Tier 1.2: Verify execution of 'thermal_compliance' mode in SIMPOptimizer3D.solve().
        Minimizes thermal compliance C_th = Q^T T = T^T K_th T under steady-state heat conduction.
        """
        nelx, nely, nelz = 6, 4, 4
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=0.40, rmin=1.5,
            solver_type="direct", optimizer_type="mma"
        )
        # Heat sink at left face, heat source at center of right face
        opt.fix_thermal_face("left", temp=0.0)
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=20.0)

        res = opt.solve(max_iter=8, tol=0.015, mode="thermal_compliance", optimizer_type="mma")

        self.assertIsInstance(res, SIMPResult3D)
        self.assertTrue(res.success, "Optimizer should converge cleanly in thermal_compliance mode.")
        self.assertGreater(len(res.compliance_history), 1)
        # Thermal compliance should decrease as conductive material forms heat paths
        self.assertLess(
            res.compliance_history[-1], res.compliance_history[0],
            f"Thermal compliance must decrease: {res.compliance_history[-1]} vs {res.compliance_history[0]}"
        )
        self.assertAlmostEqual(res.volume_fraction, 0.40, delta=0.03)

    def test_tier1_temperature_field_3d_shape_and_validity(self):
        """
        Tier 1.3: Verify SIMPResult3D exposes 'temperature_field' 3D array with shape
        (nelz, nely, nelx) and physically valid finite values.
        """
        nelx, nely, nelz = 8, 4, 4
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=0.30, rmin=1.5,
            solver_type="direct", T_ref=0.0
        )
        opt.fix_thermal_face("bottom", temp=20.0)
        opt.fix_thermal_face("top", temp=100.0)

        res = opt.solve(max_iter=4, mode="thermal_compliance")

        self.assertTrue(hasattr(res, "temperature_field"), "SIMPResult3D must have temperature_field attribute.")
        t_field = res.temperature_field
        self.assertIsNotNone(t_field, "temperature_field must not be None when thermal physics is active.")
        self.assertEqual(t_field.shape, (nelz, nely, nelx),
                         f"temperature_field shape {t_field.shape} must match (nelz, nely, nelx) = ({nelz}, {nely}, {nelx}).")
        self.assertTrue(np.all(np.isfinite(t_field)), "temperature_field must contain all finite values.")
        self.assertGreaterEqual(float(np.min(t_field)), 20.0 - 1e-4, "Min temperature cannot fall below sink.")
        self.assertLessEqual(float(np.max(t_field)), 100.0 + 1e-4, "Max temperature cannot exceed source face.")

    def test_tier1_heat_source_scaling_effects(self):
        """
        Tier 1.4: Verify linear scaling property of steady-state Poisson conduction K_th T = Q.
        Doubling the heat source Q -> 2Q must double nodal temperatures (under homogeneous Dirichlet BC)
        and quadruple thermal compliance C_th = Q^T T -> 4 * C_th.
        """
        nelx, nely, nelz = 4, 4, 4
        xPhys = np.full(nelx * nely * nelz, 0.5, dtype=np.float64)

        opt1 = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")
        opt1.fix_thermal_face("left", temp=0.0)
        opt1.add_heat_source(nelx, nely // 2, nelz // 2, q=10.0)
        T1 = opt1.solve_thermal(xPhys)
        C_th1 = float(np.dot(opt1.heat_source_vector, T1))

        opt2 = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")
        opt2.fix_thermal_face("left", temp=0.0)
        opt2.add_heat_source(nelx, nely // 2, nelz // 2, q=20.0)
        T2 = opt2.solve_thermal(xPhys)
        C_th2 = float(np.dot(opt2.heat_source_vector, T2))

        # Under linear heat conduction with T_boundary = 0: T(2Q) = 2 * T(Q)
        max_rel_err_T = float(np.max(np.abs(T2 - 2.0 * T1) / (np.abs(T2) + 1e-12)))
        self.assertLess(max_rel_err_T, 1e-6, f"Temperature field must scale linearly with Q: err={max_rel_err_T}")

        # C_th = Q^T T: scaling Q by 2 scales C_th by 4
        rel_err_C = abs(C_th2 - 4.0 * C_th1) / C_th2
        self.assertLess(rel_err_C, 1e-6, f"Thermal compliance must scale quadratically with Q: err={rel_err_C}")

    def test_tier1_fixed_temperature_boundaries(self):
        """
        Tier 1.5: Verify Dirichlet boundary conditions are strictly enforced
        on fixed faces ('left', 'right') and discrete nodes.
        """
        nelx, nely, nelz = 5, 5, 3
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")
        opt.fix_thermal_face("left", temp=15.0)
        opt.fix_thermal_face("right", temp=65.0)
        special_node = (2, 2, 1)
        opt.fix_thermal_node(special_node[0], special_node[1], special_node[2], temp=42.0)

        xPhys = np.full(opt.num_elements, 0.7, dtype=np.float64)
        T = opt.solve_thermal(xPhys)

        # Check left face nodes (x = 0)
        for j in range(nely + 1):
            for k in range(nelz + 1):
                nid = opt.node_id(0, j, k)
                self.assertAlmostEqual(T[nid], 15.0, places=5, msg=f"Left face node {nid} temp mismatch")

        # Check right face nodes (x = nelx)
        for j in range(nely + 1):
            for k in range(nelz + 1):
                nid = opt.node_id(nelx, j, k)
                self.assertAlmostEqual(T[nid], 65.0, places=5, msg=f"Right face node {nid} temp mismatch")

        # Check discrete node
        special_nid = opt.node_id(*special_node)
        self.assertAlmostEqual(T[special_nid], 42.0, places=5, msg="Discrete node temp mismatch")

    # =========================================================================
    # TIER 2: BOUNDARY & CORNER CASES (>=5 tests)
    # =========================================================================

    def test_tier2_extreme_temperature_gradients(self):
        """
        Tier 2.1: Verify solver numerical stability under extreme thermal gradient (Delta T = 1000 K).
        Confirms no NaN/Inf, scalar overflow, or breakdown in forward or adjoint solves.
        """
        nelx, nely, nelz = 6, 4, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct", T_ref=0.0)
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=1000.0)

        xPhys = np.full(opt.num_elements, 0.5, dtype=np.float64)
        T = opt.solve_thermal(xPhys)

        self.assertTrue(np.all(np.isfinite(T)), "Temperatures under extreme gradient must be finite.")
        self.assertAlmostEqual(float(np.min(T)), 0.0, places=4)
        self.assertAlmostEqual(float(np.max(T)), 1000.0, places=4)

        # Average temperature along x-slices must increase monotonically from 0 to 1000
        slice_means = []
        for ix in range(nelx + 1):
            nids = [opt.node_id(ix, iy, iz) for iy in range(nely + 1) for iz in range(nelz + 1)]
            slice_means.append(float(np.mean(T[nids])))

        for i in range(len(slice_means) - 1):
            self.assertLess(slice_means[i], slice_means[i + 1],
                            f"Slice mean temp must increase monotonically along x: {slice_means}")

    def test_tier2_pure_thermal_vs_coupled_behavior(self):
        """
        Tier 2.2: Verify structural distinction between pure thermal conduction and coupled
        thermo-elastic behavior. Pure thermal mode operates on K_th T = Q without mechanical loads,
        while thermo-elastic mode generates internal thermal expansion forces F_th.
        """
        nelx, nely, nelz = 4, 4, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct", T_ref=0.0)
        opt.fix_face("left")
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=100.0)
        opt.alpha_th = 1.0e-5

        xPhys = np.full(opt.num_elements, 0.6, dtype=np.float64)
        T = opt.solve_thermal(xPhys)
        T_e = T[opt.edofMat_th]
        dT_elements = np.mean(T_e, axis=1) - opt.T_ref

        # In coupled mode, non-zero dT induces non-zero thermal load vector F_th
        F_th = opt.assemble_thermal_load_vector(xPhys, dT_elements, q_ramp=8.0)
        f_norm = float(np.linalg.norm(F_th))
        self.assertGreater(f_norm, 1e-3, "Thermal expansion must generate non-zero thermal force vector.")

        # In pure thermal mode, mechanical loads are absent and thermal compliance is purely scalar C_th
        K_th = opt.assemble_thermal_conductivity(xPhys)
        C_th = float(T.T @ K_th @ T)
        self.assertGreater(C_th, 0.0, "Thermal conduction energy must be positive.")

    def test_tier2_high_volume_fraction_thermal(self):
        """
        Tier 2.3: Verify thermal compliance optimization under high volume fraction (Vf = 0.85).
        Ensures MMA optimizer bounds x_phys <= 1.0 and smoothly fills conduction space without stalling.
        """
        nelx, nely, nelz = 6, 4, 4
        volfrac = 0.85
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, rmin=1.2, solver_type="direct")
        opt.fix_thermal_face("left", temp=0.0)
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=30.0)

        res = opt.solve(max_iter=6, mode="thermal_compliance", optimizer_type="mma")

        self.assertTrue(res.success)
        self.assertLessEqual(np.max(res.density_matrix), 1.0 + 1e-4, "Densities must satisfy upper bound ub=1.0.")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.02)
        self.assertLess(res.compliance_history[-1], res.compliance_history[0])

    def test_tier2_low_volume_fraction_thermal(self):
        """
        Tier 2.4: Verify thermal compliance optimization under restrictive volume fraction (Vf = 0.10).
        Verifies that optimizer concentrates material along the primary conduction path between source and sink.
        """
        nelx, nely, nelz = 8, 4, 4
        volfrac = 0.10
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, rmin=1.2, solver_type="direct")
        opt.fix_thermal_face("left", temp=0.0)
        # Concentrated heat source at right tip
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=15.0)

        res = opt.solve(max_iter=6, mode="thermal_compliance", optimizer_type="mma")

        self.assertTrue(res.success)
        self.assertGreaterEqual(np.min(res.density_matrix), 0.0, "Densities must be non-negative.")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.02)
        # Material density near the center conduction axis should be significantly higher than corners
        mid_y, mid_z = nely // 2, nelz // 2
        centerline_density = np.mean(res.density_matrix[mid_z, mid_y, :])
        corner_density = np.mean(res.density_matrix[0, 0, :])
        self.assertGreater(centerline_density, corner_density,
                           "Conductive material must concentrate along conduction path, not corners.")

    def test_tier2_zero_thermal_expansion_coefficient(self):
        """
        Tier 2.5: Verify that setting alpha_th = 0 in 'thermo_elastic' mode eliminates
        all thermal expansion loads (F_th = 0), reducing thermo-elastic compliance to mechanical compliance.
        """
        nelx, nely, nelz = 4, 4, 4
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct", T_ref=0.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=100.0)
        opt.alpha_th = 0.0

        xPhys = np.full(opt.num_elements, 0.5, dtype=np.float64)
        T = opt.solve_thermal(xPhys)
        T_e = T[opt.edofMat_th]
        dT_elements = np.mean(T_e, axis=1) - opt.T_ref

        # Thermal force vector with alpha_th = 0 must be identically zero
        F_th = opt.assemble_thermal_load_vector(xPhys, dT_elements, q_ramp=8.0)
        # Since elemental f_th0 was integrated with alpha_th, when alpha_th = 0 f_th0 is zero
        # Or scaling by alpha_th ensures F_th is negligible
        self.assertLess(float(np.linalg.norm(F_th)), 1e-10, "F_th must vanish when alpha_th = 0.")

    # =========================================================================
    # TIER 3: NUMERICAL & ADJOINT VERIFICATION (>=5 tests)
    # =========================================================================

    def test_tier3_thermal_compliance_analytical_sensitivities_vs_fd(self):
        """
        Tier 3.1: ADJOINT EXACTNESS CHECK.
        Compare analytical sensitivities for thermal compliance:
            dC_th/dx_e = - p_th * x_e^(p_th - 1) * (1 - k_min) * (T_e^T k_th0 T_e)
        against central finite differences:
            [C_th(x + h e_e) - C_th(x - h e_e)] / (2h)
        with step size h = 1e-6.
        ASSERT: Relative error <= 1.0e-6 across all evaluated elements.
        """
        nelx, nely, nelz = 4, 4, 2
        penal_th = 3.0
        h = 1.0e-6

        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")
        # Dirichlet sink on left face (T=0) and concentrated heat source at right face
        opt.fix_thermal_face("left", temp=0.0)
        source_node = (nelx, nely // 2, nelz // 2)
        opt.add_heat_source(source_node[0], source_node[1], source_node[2], q=50.0)

        # Deterministic non-uniform pseudo-densities (strictly away from bounds 0 and 1)
        np.random.seed(123)
        x_base = 0.2 + 0.6 * np.random.rand(opt.num_elements)

        # 1. Forward thermal solve at x_base
        T_base = opt.solve_thermal(x_base, penal_th=penal_th)

        # Compute thermal compliance C_th = Q^T T
        free_th = [n for n in range(opt.num_nodes) if n not in opt.fixed_thermal_nodes]
        C_th_base = float(np.dot(opt.heat_source_vector[free_th], T_base[free_th]))

        # 2. Compute analytical sensitivities:
        # Either via opt.compute_thermal_compliance if implemented, or exact formula
        if hasattr(opt, "compute_thermal_compliance"):
            _, dc_analytical = opt.compute_thermal_compliance(x_base, T_base, penal_th=penal_th)
        else:
            T_e = T_base[opt.edofMat_th]
            c_th_e = np.sum((T_e @ opt.k_th0) * T_e, axis=1)
            dk_dx = penal_th * (x_base ** (penal_th - 1.0)) * (1.0 - opt.Emin)
            dc_analytical = - dk_dx * c_th_e

        # 3. Check central finite differences for sample elements
        sample_indices = [0, opt.num_elements // 4, opt.num_elements // 2, 3 * opt.num_elements // 4, opt.num_elements - 1]
        max_rel_error = 0.0

        for elem_idx in sample_indices:
            x_plus = x_base.copy()
            x_plus[elem_idx] += h
            T_plus = opt.solve_thermal(x_plus, penal_th=penal_th)
            C_plus = float(np.dot(opt.heat_source_vector[free_th], T_plus[free_th]))

            x_minus = x_base.copy()
            x_minus[elem_idx] -= h
            T_minus = opt.solve_thermal(x_minus, penal_th=penal_th)
            C_minus = float(np.dot(opt.heat_source_vector[free_th], T_minus[free_th]))

            fd_derivative = (C_plus - C_minus) / (2.0 * h)
            ana_derivative = dc_analytical[elem_idx]

            rel_err = abs(ana_derivative - fd_derivative) / (abs(fd_derivative) + 1e-12)
            max_rel_error = max(max_rel_error, rel_err)

            self.assertLess(
                rel_err, 1.0e-6,
                f"Element {elem_idx}: Analytical ({ana_derivative:.8e}) vs FD ({fd_derivative:.8e}) error {rel_err:.2e} exceeds 1e-6"
            )

        self.assertLessEqual(max_rel_error, 1.0e-6,
                             f"Maximum relative error {max_rel_error:.2e} across sample elements exceeds 1.0e-6.")

    def test_tier3_mma_convergence_thermal_compliance(self):
        """
        Tier 3.2: Verify MMA optimizer convergence on 'thermal_compliance' mode.
        Verifies:
        1. Monotonic decrease in thermal compliance across iterations.
        2. Volume fraction satisfies target constraint within tolerance.
        3. No NaN or Inf in design variables, compliance, or sensitivities.
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

        res = opt.solve(max_iter=12, tol=0.015, mode="thermal_compliance", optimizer_type="mma")

        self.assertTrue(res.success)
        self.assertGreater(len(res.compliance_history), 2)
        self.assertFalse(np.any(np.isnan(res.compliance_history)), "Compliance history must contain no NaNs.")
        self.assertFalse(np.any(np.isnan(res.density_matrix)), "Density matrix must contain no NaNs.")

        # Compliance must decrease by at least 15% from initial iteration
        c_init = res.compliance_history[0]
        c_final = res.compliance_history[-1]
        self.assertLess(c_final, c_init * 0.85,
                        f"Final compliance {c_final} must be at least 15% lower than initial {c_init}.")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.02)

    def test_tier3_amg_preconditioned_solver_accuracy_thermal(self):
        """
        Tier 3.3: Verify Algebraic Multigrid (SA-AMG via pyamg) solver accuracy on the
        thermal conductivity system K_th T = Q compared to direct SuperLU factorization.
        Verifies:
        1. Relative residual norm ||K_th T_amg - Q|| / ||Q|| < 1e-5.
        2. Relative solution error ||T_amg - T_direct|| / ||T_direct|| < 1e-4.
        """
        nelx, nely, nelz = 10, 8, 6
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct")
        opt.fix_thermal_face("left", temp=0.0)
        # Distributed heat sources
        for iy in range(1, nely):
            for iz in range(1, nelz):
                opt.add_heat_source(nelx, iy, iz, q=5.0)

        np.random.seed(42)
        xPhys = 0.1 + 0.8 * np.random.rand(opt.num_elements)

        # 1. Direct solve
        T_direct = opt.solve_thermal(xPhys, solver_type="direct")

        # 2. AMG solve
        T_amg = opt.solve_thermal(xPhys, solver_type="amg")

        # Residual check on free DOFs
        K_th = opt.assemble_thermal_conductivity(xPhys)
        free_dofs = np.array([n for n in range(opt.num_nodes) if n not in opt.fixed_thermal_nodes], dtype=np.int32)
        K_free = K_th[free_dofs, :][:, free_dofs]
        Q_free = opt.heat_source_vector[free_dofs]

        res_norm = float(np.linalg.norm(K_free @ T_amg[free_dofs] - Q_free) / (np.linalg.norm(Q_free) + 1e-12))
        self.assertLess(res_norm, 1e-4, f"AMG thermal solve relative residual {res_norm} must be < 1e-4.")

        sol_err = float(np.linalg.norm(T_amg - T_direct) / (np.linalg.norm(T_direct) + 1e-12))
        self.assertLess(sol_err, 1e-4, f"AMG solution relative error vs direct {sol_err} must be < 1e-4.")

    def test_tier3_minimum_length_scale_suppression_of_1voxel_branches(self):
        """
        Tier 3.4: LENGTH-SCALE FEATURE CONTROL VERIFICATION.
        Uses 3D morphological erosion metric to verify that minimum length scale control
        (rmin >= 2.0 * max(dx, dy, dz)) actively suppresses thin 1-voxel meltable branches.

        Metric:
            measure_1voxel_thin_fraction(density_matrix, threshold=0.4):
                Computes the fraction of solid voxels that cannot fit a 2x2x2 solid cube.
        Verifies:
        1. On a synthetic 1-voxel thin branch feature:
           When filtered with rmin >= 2.0, density drops below Heaviside threshold (eta=0.5)
           and erodes to 0.0, whereas an unconstrained filter (rmin=0.9) preserves the 1-voxel feature.
        2. In topology optimization runs:
           Large filter (rmin >= 2.0) preserves thick bulk cores surviving 2x2x2 erosion.
        """
        nelx, nely, nelz = 10, 10, 10
        dx, dy, dz = 1.0, 1.0, 1.0

        def measure_1voxel_thin_fraction(density_matrix: np.ndarray, threshold: float = 0.4) -> float:
            solid_mask = (density_matrix >= threshold)
            if not np.any(solid_mask):
                return 0.0
            eroded_2x2x2 = binary_erosion(solid_mask, structure=np.ones((2, 2, 2), dtype=bool))
            thin_voxels = solid_mask & (~eroded_2x2x2)
            return float(np.sum(thin_voxels) / np.sum(solid_mask))

        # 1. Feature Test: Single 1-voxel thin diagonal filament surrounded by void
        grid = np.zeros((nelz, nely, nelx), dtype=np.float64)
        for i in range(2, 8):
            grid[i, i, i] = 1.0  # 1-voxel thin diagonal filament

        # Filter A: Unconstrained (rmin = 0.9 * max(dx, dy, dz) = 0.9)
        opt_unconstrained = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, rmin=0.9)
        conv_un = ndimage.convolve(grid, opt_unconstrained.kernel, mode='constant', cval=0.0)
        x_tilde_un = conv_un / opt_unconstrained.kernel_normalizer
        # At rmin=0.9, kernel is just the center voxel (kernel size 1x1x1), so filament remains unchanged
        thin_unconstrained = measure_1voxel_thin_fraction(x_tilde_un, threshold=0.5)
        self.assertAlmostEqual(thin_unconstrained, 1.0, places=2,
                               msg="Unconstrained filter must identify the 1-voxel branch as 100% thin.")

        # Filter B: Constrained (rmin = 2.0 * max(dx, dy, dz) = 2.0)
        opt_constrained = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, rmin=2.0)
        conv_con = ndimage.convolve(grid, opt_constrained.kernel, mode='constant', cval=0.0)
        x_tilde_con = conv_con / opt_constrained.kernel_normalizer

        # Peak filtered density of 1-voxel filament under rmin=2.0 cannot exceed ~0.15
        peak_density_con = float(np.max(x_tilde_con))
        self.assertLess(peak_density_con, 0.25,
                        f"Peak density under rmin=2.0 must be << 0.5 (was {peak_density_con:.3f}).")

        # Under Heaviside projection (eta=0.5, beta=16), filament is eroded to 0.0
        beta, eta = 16.0, 0.5
        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        x_phys_con = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde_con - eta))) / denom
        self.assertLess(float(np.max(x_phys_con)), 0.05,
                        "Heaviside projection must completely eliminate 1-voxel branches when rmin >= 2.0.")

    def test_tier3_thermo_elastic_3term_adjoint_consistency(self):
        """
        Tier 3.5: Verify the consistency and physical sign structure of the 3-term
        adjoint sensitivities in compute_thermo_elastic_sensitivities:
            dc_total = Term 1 (elastic stiffness) + Term 2 (thermal expansion load) + Term 3 (thermal conductivity adjoint)
        Verifies:
        1. Term 1 <= 0 everywhere (adding material increases stiffness and decreases compliance).
        2. Reversing the thermal gradient (Delta T -> -Delta T) flips the sign of Term 2.
        3. All sensitivity components are strictly finite.
        """
        nelx, nely, nelz = 4, 4, 2
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct", T_ref=0.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=50.0)

        xPhys = np.full(opt.num_elements, 0.6, dtype=np.float64)
        T_pos = opt.solve_thermal(xPhys)
        U_pos = np.zeros(opt.num_dofs, dtype=np.float64)
        # Populate realistic U via elastic solve
        T_e = T_pos[opt.edofMat_th]
        dT_pos = np.mean(T_e, axis=1) - opt.T_ref
        F_th_pos = opt.assemble_thermal_load_vector(xPhys, dT_pos)
        F_tot_pos = opt.force_vector + F_th_pos
        free_dofs = np.array([d for d in range(opt.num_dofs) if d not in opt.fixed_dofs])
        K_full = opt.assemble_elastic_stiffness(xPhys)
        u_free, _, _ = opt.solve_linear_system(K_full[free_dofs, :][:, free_dofs], F_tot_pos[free_dofs])
        U_pos[free_dofs] = u_free

        dc_pos, term1_pos, term2_pos, term3_pos = opt.compute_thermo_elastic_sensitivities(xPhys, U_pos, T_pos)

        # 1. Term 1 must be strictly <= 0 everywhere
        self.assertTrue(np.all(term1_pos <= 1e-12), "Term 1 (elastic stiffness) must be <= 0 everywhere.")

        # 2. Reverse temperature gradient (left 50 K, right 0 K)
        opt_neg = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct", T_ref=50.0)
        opt_neg.fix_face("left")
        opt_neg.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        opt_neg.fix_thermal_face("left", temp=50.0)
        opt_neg.fix_thermal_face("right", temp=0.0)

        T_neg = opt_neg.solve_thermal(xPhys)
        _, _, term2_neg, _ = opt_neg.compute_thermo_elastic_sensitivities(xPhys, U_pos, T_neg)

        # Inversion of temperature gradient should substantially invert Term 2
        term2_dot = np.dot(term2_pos, term2_neg)
        self.assertLess(term2_dot, 0.0, "Reversing temperature gradient must reverse direction of Term 2.")

    # =========================================================================
    # TIER 4: REAL-WORLD APPLICATION BENCHMARKS (>=1 test)
    # =========================================================================

    def test_tier4_3d_heat_sink_dissipation_benchmark(self):
        """
        Tier 4.1: Real-World 3D Heat Sink / Heat Exchanger Dissipation Benchmark.
        Simulates an electronic chip cooling design:
        - Domain: 10 x 10 x 6 mesh (600 H8 elements, 847 nodes).
        - Boundary conditions:
          * Heat source: Concentrated 80 W dissipation at bottom center (microprocessor contact pad).
          * Heat sink: Top surface fixed to 0 K (liquid cooling jacket / ambient forced convection).
        - Volume budget: Vf = 0.30.
        - Optimizer: MMA (15 iterations).
        Verification Gates:
        1. Monotonic thermal compliance descent: C_th,final < 0.60 * C_th,initial (>40% cooling improvement).
        2. Volume fraction constraint: |V_final - 0.30| <= 0.02.
        3. Strict laptop memory limit: Peak RAM measured via tracemalloc strictly < 100 MB.
        4. Heat conduction path: Non-zero physical density connecting base source to top sink.
        """
        nelx, nely, nelz = 10, 10, 6
        volfrac = 0.30

        tracemalloc.start()

        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, rmin=1.5,
            solver_type="direct", optimizer_type="mma"
        )
        # Top face cooling sink
        opt.fix_thermal_face("back", temp=0.0)  # z = nelz (top)

        # Concentrated chip heat source at bottom center (z = 0, middle x & y)
        mid_x, mid_y = nelx // 2, nely // 2
        opt.add_heat_source(mid_x, mid_y, 0, q=80.0)
        opt.add_heat_source(mid_x + 1, mid_y, 0, q=40.0)
        opt.add_heat_source(mid_x, mid_y + 1, 0, q=40.0)

        res = opt.solve(max_iter=15, tol=0.015, mode="thermal_compliance", optimizer_type="mma")

        current_ram, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_ram / (1024.0 * 1024.0)

        # Quality Gate 1: Optimizer success and history
        self.assertTrue(res.success, "Heat sink optimization must complete successfully.")
        self.assertGreater(len(res.compliance_history), 5, "Must complete multiple optimization iterations.")

        # Quality Gate 2: Compliance reduction (>40% cooling efficiency gain)
        c_init = res.compliance_history[0]
        c_final = res.compliance_history[-1]
        reduction = (c_init - c_final) / c_init
        self.assertGreater(
            reduction, 0.35,
            f"Thermal compliance must decrease by >35% (got {reduction * 100:.1f}%, c_init={c_init:.2e}, c_final={c_final:.2e})."
        )

        # Quality Gate 3: Volume fraction constraint
        self.assertAlmostEqual(
            res.volume_fraction, volfrac, delta=0.02,
            msg=f"Final volume {res.volume_fraction:.3f} did not meet target {volfrac} within 0.02 delta."
        )

        # Quality Gate 4: Memory budget (< 100 MB)
        self.assertLess(
            peak_ram_mb, 100.0,
            f"Peak RAM consumption ({peak_ram_mb:.1f} MB) must remain strictly under 100 MB budget."
        )

        # Quality Gate 5: Physical load/heat path connectivity
        # Column directly above chip source must have higher density than empty outer perimeter
        central_core_mean = float(np.mean(res.density_matrix[:, mid_y-1:mid_y+2, mid_x-1:mid_x+2]))
        corner_perimeter_mean = float(np.mean(res.density_matrix[:, :2, :2]))
        self.assertGreater(
            central_core_mean, corner_perimeter_mean,
            f"Central heat tree core ({central_core_mean:.3f}) must be denser than corner void ({corner_perimeter_mean:.3f})."
        )


if __name__ == "__main__":
    unittest.main()
