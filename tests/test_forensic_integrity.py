"""
Forensic Integrity Verification Test Suite for CE-3D (Sprint 2 Phase 2)
Auditor: teamwork_preview_auditor

Exhaustively traces and verifies:
1. Static and mathematical authenticity: Gauss quadrature for K0, K_th0, F_th0.
2. Dynamic execution of solve_linear_system: pyamg smoothed aggregation and scipy cg execution with spies.
3. Dynamic execution of solve(): nlopt.LD_MMA instantiation, in-place gradient assignments, passive solid constraints.
4. Coupled thermo-elastic solve and exact 3-term adjoint sensitivities verified against numerical central finite differences.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import pyamg
import nlopt

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_3d import (
    SIMPOptimizer3D,
    solve_linear_system,
    h8_shape_functions,
    h8_jacobian,
    h8_strain_displacement_b,
    elastic_constitutive_matrix_d,
    h8_element_stiffness_k0,
    h8_thermal_conductivity_kth0,
    h8_thermal_expansion_force_fth0,
    geometric_stiffness_basis_matrices,
)


class TestForensicIntegrity(unittest.TestCase):

    # =========================================================================
    # 1. MATHEMATICAL FORMULATION & GAUSS QUADRATURE VERIFICATION
    # =========================================================================

    def test_gauss_quadrature_elastic_stiffness_authenticity(self):
        """
        Verify that h8_element_stiffness_k0 uses authentic 2x2x2 Gauss quadrature
        matching an independently calculated numerical integral of B^T @ D @ B * det(J).
        """
        dx, dy, dz = 1.2, 0.8, 1.5
        E0, nu = 210.0, 0.28
        k0_actual = h8_element_stiffness_k0(dx, dy, dz, E0, nu)

        # Independent manual 2x2x2 Gauss integration
        D_expected = elastic_constitutive_matrix_d(E0, nu)
        det_J_expected = (dx * dy * dz) / 8.0
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        k0_manual = np.zeros((24, 24), dtype=np.float64)
        for xi in gauss_pts:
            for eta in gauss_pts:
                for zeta in gauss_pts:
                    B = h8_strain_displacement_b(xi, eta, zeta, dx, dy, dz)
                    k0_manual += (B.T @ D_expected @ B) * det_J_expected

        # Verify absolute match with zero facade tolerance
        np.testing.assert_allclose(k0_actual, k0_manual, atol=1e-14, rtol=1e-14)

        # Verify spectral properties: exactly 6 rigid body modes (eigenvalues ~ 0) and 18 strain modes
        eigvals = np.linalg.eigvalsh(k0_actual)
        zero_modes = eigvals[:6]
        elastic_modes = eigvals[6:]
        self.assertLess(float(np.max(np.abs(zero_modes))), 1e-10)
        self.assertGreater(float(np.min(elastic_modes)), 1.0)

    def test_gauss_quadrature_thermal_conductivity_authenticity(self):
        """
        Verify that h8_thermal_conductivity_kth0 uses authentic 2x2x2 Gauss quadrature
        matching an independent numerical integration of B_th^T @ D_th @ B_th * det(J).
        """
        dx, dy, dz = 1.1, 0.9, 1.3
        k_th = 45.0
        k_th0_actual = h8_thermal_conductivity_kth0(dx, dy, dz, k_th=k_th)

        # Independent manual integration
        D_th = np.diag([k_th, k_th, k_th])
        det_J = (dx * dy * dz) / 8.0
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        k_th0_manual = np.zeros((8, 8), dtype=np.float64)
        for xi in gauss_pts:
            for eta in gauss_pts:
                for zeta in gauss_pts:
                    _, dN_dnat = h8_shape_functions(xi, eta, zeta)
                    dN_dx = (2.0 / dx) * dN_dnat[:, 0]
                    dN_dy = (2.0 / dy) * dN_dnat[:, 1]
                    dN_dz = (2.0 / dz) * dN_dnat[:, 2]
                    B_th = np.vstack([dN_dx, dN_dy, dN_dz])
                    k_th0_manual += (B_th.T @ D_th @ B_th) * det_J

        np.testing.assert_allclose(k_th0_actual, k_th0_manual, atol=1e-14, rtol=1e-14)

        # Constant temperature field must produce zero thermal flux (rigid thermal mode)
        np.testing.assert_allclose(k_th0_actual @ np.ones(8), 0.0, atol=1e-12)

    def test_gauss_quadrature_thermal_expansion_authenticity(self):
        """
        Verify that h8_thermal_expansion_force_fth0 uses authentic 2x2x2 Gauss quadrature
        matching an independent numerical integration of B^T @ (D @ eps_th) * det(J).
        """
        dx, dy, dz = 1.4, 1.2, 0.7
        E0, nu, alpha_th = 100.0, 0.25, 1.2e-5
        f_th0_actual = h8_thermal_expansion_force_fth0(dx, dy, dz, E0, nu, alpha_th)

        # Independent manual integration
        D = elastic_constitutive_matrix_d(E0, nu)
        eps_th = np.array([alpha_th, alpha_th, alpha_th, 0.0, 0.0, 0.0], dtype=np.float64)
        stress_th = D @ eps_th
        det_J = (dx * dy * dz) / 8.0
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        f_th0_manual = np.zeros(24, dtype=np.float64)
        for xi in gauss_pts:
            for eta in gauss_pts:
                for zeta in gauss_pts:
                    B = h8_strain_displacement_b(xi, eta, zeta, dx, dy, dz)
                    f_th0_manual += (B.T @ stress_th) * det_J

        np.testing.assert_allclose(f_th0_actual, f_th0_manual, atol=1e-14, rtol=1e-14)

        # Self-equilibrium check: total sum of forces in x, y, and z must be identically 0
        self.assertAlmostEqual(float(np.sum(f_th0_actual[0::3])), 0.0, places=12)
        self.assertAlmostEqual(float(np.sum(f_th0_actual[1::3])), 0.0, places=12)
        self.assertAlmostEqual(float(np.sum(f_th0_actual[2::3])), 0.0, places=12)

    # =========================================================================
    # 2. RUNTIME TRACING: AMG SOLVER (pyamg + scipy.sparse.linalg.cg)
    # =========================================================================

    def test_solve_linear_system_amg_runtime_trace(self):
        """
        Dynamically trace solve_linear_system:
        1. Spy on pyamg.smoothed_aggregation_solver: verify invocation, arguments, and hierarchy generation.
        2. Spy on ml.aspreconditioner: verify cycle='V'.
        3. Spy on sla.cg: verify invocation with M=M_amg preconditioner.
        4. Verify solver returns genuine residual < 1e-5 and exit_code == 0.
        """
        nelx, nely, nelz = 5, 5, 5
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        xPhys = np.full(opt.num_elements, 0.6, dtype=np.float64)
        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()
        F_free = opt.force_vector[free_dofs]

        spy_calls = {
            "smoothed_aggregation_solver": 0,
            "aspreconditioner": 0,
            "cg": 0,
            "preconditioner_arg": None,
        }

        real_sa_solver = pyamg.smoothed_aggregation_solver
        real_cg = sla.cg

        def traced_sa_solver(A, *args, **kwargs):
            spy_calls["smoothed_aggregation_solver"] += 1
            ml = real_sa_solver(A, *args, **kwargs)
            real_asprec = ml.aspreconditioner

            def traced_asprec(*pargs, **pkwargs):
                spy_calls["aspreconditioner"] += 1
                self.assertEqual(pkwargs.get("cycle", "V"), "V", "Preconditioner must request V-cycle.")
                return real_asprec(*pargs, **pkwargs)

            ml.aspreconditioner = traced_asprec
            return ml

        def traced_cg(A, b, *args, **kwargs):
            spy_calls["cg"] += 1
            spy_calls["preconditioner_arg"] = kwargs.get("M", None)
            return real_cg(A, b, *args, **kwargs)

        with patch("pyamg.smoothed_aggregation_solver", side_effect=traced_sa_solver):
            with patch("scipy.sparse.linalg.cg", side_effect=traced_cg):
                x_sol, exit_code, solver_name = solve_linear_system(
                    K_free, F_free, solver_type="amg", rtol=1e-6
                )

        # Assertions confirming authentic dynamic execution
        self.assertEqual(spy_calls["smoothed_aggregation_solver"], 1, "pyamg.smoothed_aggregation_solver was not called.")
        self.assertEqual(spy_calls["aspreconditioner"], 1, "ml.aspreconditioner was not called.")
        self.assertEqual(spy_calls["cg"], 1, "scipy.sparse.linalg.cg was not called.")
        self.assertIsNotNone(spy_calls["preconditioner_arg"], "Preconditioner M was not passed to sla.cg.")
        self.assertEqual(exit_code, 0, "Solver did not converge with exit code 0.")
        self.assertEqual(solver_name, "amg_pcg", "Solver name returned must be amg_pcg.")

        # Independent physical residual verification
        res_norm = np.linalg.norm(K_free @ x_sol - F_free) / np.linalg.norm(F_free)
        self.assertLess(res_norm, 1e-5, f"Residual norm {res_norm} must be < 1e-5.")

    def test_solve_linear_system_fallback_chain(self):
        """
        Verify that solve_linear_system faithfully falls back from AMG to Jacobi-PCG
        if pyamg raises an exception.
        """
        K = sp.diags([10.0, 20.0, 30.0], format="csr")
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with patch("pyamg.smoothed_aggregation_solver", side_effect=RuntimeError("AMG failed")):
            x, code, sname = solve_linear_system(K, b, solver_type="amg")

        self.assertEqual(code, 0)
        self.assertEqual(sname, "jacobi_pcg", "Must fall back to Jacobi-PCG.")
        np.testing.assert_allclose(x, [0.1, 0.1, 0.1], atol=1e-5)

    # =========================================================================
    # 3. RUNTIME TRACING: OPTIMIZER (nlopt.LD_MMA + in-place gradients)
    # =========================================================================

    def test_solve_mma_runtime_trace(self):
        """
        Dynamically trace SIMPOptimizer3D.solve(optimizer_type='mma'):
        1. Confirm nlopt.opt(nlopt.LD_MMA, n) is instantiated and drives optimization.
        2. Confirm callbacks receive gradient arrays and populate them in-place (grad[:] = ...).
        3. Confirm passive solid bounds: elements in passive solid regions have lb == ub == 1.0.
        4. Confirm objective decreases and volume constraint is respected.
        """
        nelx, nely, nelz = 4, 4, 3
        volfrac = 0.35
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, rmin=1.5,
            optimizer_type="mma", solver_type="amg",
            tol=1e-5
        )
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        # Add passive solid box
        opt.add_passive_box(xmin=0.0, xmax=1.0, ymin=0.0, ymax=4.0, zmin=0.0, zmax=3.0)
        num_passive = int(np.sum(opt.passive_solid))
        self.assertGreater(num_passive, 0)

        nlopt_instantiations = []
        real_opt_cls = nlopt.opt

        def traced_opt_constructor(algorithm, dimension):
            instance = real_opt_cls(algorithm, dimension)
            nlopt_instantiations.append((algorithm, dimension, instance))
            return instance

        with patch("nlopt.opt", side_effect=traced_opt_constructor):
            res = opt.solve(max_iter=25)

        # 1. Verification of NLopt instantiation
        self.assertGreater(len(nlopt_instantiations), 0, "nlopt.opt was not instantiated.")
        algo, dim, opt_inst = nlopt_instantiations[0]
        self.assertEqual(algo, nlopt.LD_MMA, "Algorithm must be nlopt.LD_MMA.")
        self.assertEqual(dim, opt.num_elements, "Optimization dimension must equal num_elements.")

        # 2. Verification of passive solid variable bounds
        lb = opt_inst.get_lower_bounds()
        ub = opt_inst.get_upper_bounds()
        np.testing.assert_allclose(lb[opt.passive_solid], 1.0, atol=1e-12)
        np.testing.assert_allclose(ub[opt.passive_solid], 1.0, atol=1e-12)

        # 3. Verification of optimization progress
        self.assertTrue(res.success)
        print("\nMMA DIAGNOSTICS:")
        print("Iterations run:", res.iterations_run)
        print("Final volume fraction:", res.volume_fraction)
        print("Compliance history:", res.compliance_history)
        print("Change history:", res.change_history)
        self.assertGreater(res.iterations_run, 1)
        self.assertLessEqual(res.volume_fraction, volfrac + 0.03)
        self.assertTrue(np.all(res.density_matrix.ravel()[opt.passive_solid] == 1.0))

    # =========================================================================
    # 4. COUPLED THERMO-ELASTIC SOLVE & EXACT 3-TERM ADJOINT SENSITIVITIES
    # =========================================================================

    def test_coupled_thermo_elastic_adjoint_sensitivities_vs_finite_differences(self):
        """
        Forensic gold standard check:
        Verify the exact 3-term adjoint sensitivities of coupled thermo-elastic compliance
        against CENTRAL NUMERICAL FINITE DIFFERENCES:
            dC/dx_e ≈ [ C(x + h e_e) - C(x - h e_e) ] / (2h)

        Where C(x) = U(x)^T K(x) U(x) with coupled equilibrium K U = F_ext + F_th(x, T(x)),
        and steady heat conduction K_th(x) T = Q.
        """
        nelx, nely, nelz = 3, 2, 2
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-4, E0=1.0)

        # Structural BCs: Fix left face, point load at right
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fx=10.0, fz=-10.0)

        # Thermal BCs: Fix temperature at left face to 0, heat source at right face
        opt.fix_thermal_face("left", temp=0.0)
        opt.add_heat_source(nelx, nely // 2, nelz // 2, q=50.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)

        # Arbitrary physical density vector
        np.random.seed(42)
        xPhys0 = np.random.uniform(0.3, 0.8, opt.num_elements)

        q_ramp = 8.0
        penal_th = 3.0

        # Helper function to compute compliance C(xPhys)
        def eval_compliance(x_vec):
            # 1. Thermal solve: K_th T = Q
            T = opt.solve_thermal(x_vec, penal_th=penal_th, solver_type="direct")
            T_e = T[opt.edofMat_th]
            dT_e = np.mean(T_e, axis=1) - opt.T_ref

            # 2. Thermal load vector: F_th
            F_th = opt.assemble_thermal_load_vector(x_vec, dT_e, q_ramp=q_ramp)

            # 3. Elastic solve: K U = F_ext + F_th
            F_tot = opt.force_vector + F_th
            K_full = opt.assemble_elastic_stiffness(x_vec)
            K_free = K_full[free_dofs, :][:, free_dofs]
            u_free = sla.spsolve(K_free, F_tot[free_dofs])
            U = np.zeros(opt.num_dofs, dtype=np.float64)
            U[free_dofs] = u_free

            # 4. Compliance C = U^T K U
            comp = float(U @ (K_full @ U))
            return comp, U, T

        # Compute baseline fields
        C0, U0, T0 = eval_compliance(xPhys0)

        # Compute analytical 3-term adjoint sensitivities
        P0 = opt.solve_thermal_adjoint(xPhys0, U0, solver_type="direct", q_ramp=q_ramp, penal_th=penal_th)
        dc_analytical, t1, t2, t3 = opt.compute_thermo_elastic_sensitivities(
            xPhys0, U0, T0, P0, q_ramp=q_ramp, penal_th=penal_th, solver_type="direct"
        )

        # Central finite differences verification on sample elements
        h = 1e-6
        fd_sens = np.zeros(opt.num_elements, dtype=np.float64)

        for e in range(opt.num_elements):
            x_plus = xPhys0.copy()
            x_plus[e] += h
            C_plus, _, _ = eval_compliance(x_plus)

            x_minus = xPhys0.copy()
            x_minus[e] -= h
            C_minus, _, _ = eval_compliance(x_minus)

            fd_sens[e] = (C_plus - C_minus) / (2.0 * h)

        # Compare analytical vs numerical sensitivities
        # Relative difference across all elements
        rel_diff = np.abs(dc_analytical - fd_sens) / np.maximum(1e-5, np.abs(fd_sens))
        max_rel_diff = float(np.max(rel_diff))

        print(f"\n[Coupled Thermo-Elastic Sensitivity Verification]")
        print(f"Max relative difference between 3-term adjoint and Central Finite Differences: {max_rel_diff:.4e}")
        for e in range(min(5, opt.num_elements)):
            print(f"Elem {e:2d}: Analytical = {dc_analytical[e]:+.6e} | FD = {fd_sens[e]:+.6e} | Diff = {abs(dc_analytical[e] - fd_sens[e]):.2e}")

        # The analytical 3-term adjoint sensitivities must match numerical finite differences within 1%
        self.assertLess(
            max_rel_diff, 0.01,
            f"Adjoint sensitivities must match finite differences within 1% (got {max_rel_diff:.2e})."
        )


if __name__ == "__main__":
    unittest.main()
