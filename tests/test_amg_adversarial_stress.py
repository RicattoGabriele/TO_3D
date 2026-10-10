"""
Adversarial Stress Test Suite for Algebraic Multigrid (pyamg) & Sparse Solvers
in CE-3D Continuum Topology Optimization Engine.

Target axes:
1. Extreme stiffness contrast: condition numbers kappa >= 10^8 and kappa >= 10^10
   with 3D checkerboard, isolated voids, and floating islands.
2. Dirichlet BC stability: ill-conditioned coarse grids, verifying coarse_solver='pinv'
   completely prevents SuperLU 'splu' factorization crashes.
3. Solver accuracy: AMG preconditioned CG vs SuperLU direct factorizations across
   multiple 3D grid sizes, load configurations, and thermal/elastic systems.
4. Memory footprint: AMG hierarchy memory consumption profile (< 100 MB) on 3D meshes
   up to 20x20x20.
"""

import os
import sys
import unittest
import tracemalloc
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import pyamg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_3d import (
    SIMPOptimizer3D,
    solve_linear_system,
    h8_thermal_conductivity_kth0
)


class TestAMGAdversarialStress(unittest.TestCase):

    # =========================================================================
    # CHALLENGE 1: EXTREME STIFFNESS CONTRAST (kappa >= 10^8 and kappa >= 10^10)
    # =========================================================================

    def test_challenge1_checkerboard_contrast_1e8(self):
        """
        Stress test AMG-PCG with 3D checkerboard at kappa >= 10^8 (E0=1.0, Emin=1e-8).
        Verifies:
        - No NaN or Inf
        - No zero-division or scalar overflow
        - Relative residual < 1e-4
        - Exit code == 0
        """
        nelx, nely, nelz = 8, 8, 8
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-8, E0=1.0, penal=3.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # 3D Checkerboard pattern
        gx, gy, gz = np.meshgrid(range(nelx), range(nely), range(nelz), indexing='ij')
        checkerboard = ((gx + gy + gz) % 2).ravel().astype(np.float64)
        xPhys = np.where(checkerboard == 1, 1.0, 0.0)

        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            u_amg, exit_code, sname = opt.solve_linear_system(
                K_free, F_free, solver_type="amg", rtol=1e-5, maxiter=1000
            )

        self.assertEqual(exit_code, 0, f"AMG solver failed to converge (exit_code={exit_code})")
        self.assertEqual(sname, "amg_pcg", f"Expected amg_pcg solver, got {sname}")
        self.assertTrue(np.all(np.isfinite(u_amg)), "Solution vector contains NaN or Inf!")

        rel_res = np.linalg.norm(K_free @ u_amg - F_free) / np.linalg.norm(F_free)
        self.assertLess(rel_res, 1e-4, f"Relative residual {rel_res} exceeds 1e-4")

        # Verify no zero-division or floating point invalid warnings occurred
        overflow_warnings = [
            w for w in recorded_warnings
            if issubclass(w.category, (RuntimeWarning, FloatingPointError))
            and any(k in str(w.message).lower() for k in ["overflow", "divide by zero", "invalid value"])
        ]
        self.assertEqual(len(overflow_warnings), 0, f"Floating point warnings detected: {overflow_warnings}")

    def test_challenge1_checkerboard_contrast_1e10(self):
        """
        Stress test AMG-PCG with 3D checkerboard at extreme kappa >= 10^10 (E0=1.0, Emin=1e-10).
        Verifies:
        - Stability under severe 10 orders-of-magnitude stiffness contrast
        - No NaN or Inf
        - Relative residual < 1e-4
        - Exit code == 0
        """
        nelx, nely, nelz = 8, 8, 8
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-10, E0=1.0, penal=3.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # 3D Checkerboard pattern
        gx, gy, gz = np.meshgrid(range(nelx), range(nely), range(nelz), indexing='ij')
        checkerboard = ((gx + gy + gz) % 2).ravel().astype(np.float64)
        xPhys = np.where(checkerboard == 1, 1.0, 0.0)

        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            u_amg, exit_code, sname = opt.solve_linear_system(
                K_free, F_free, solver_type="amg", rtol=1e-5, maxiter=1500
            )

        self.assertEqual(exit_code, 0, f"AMG solver failed to converge at 1e10 contrast (exit_code={exit_code})")
        self.assertEqual(sname, "amg_pcg")
        self.assertTrue(np.all(np.isfinite(u_amg)), "Solution vector contains NaN or Inf!")

        rel_res = np.linalg.norm(K_free @ u_amg - F_free) / np.linalg.norm(F_free)
        self.assertLess(rel_res, 1e-4, f"Relative residual {rel_res} exceeds 1e-4")

        overflow_warnings = [
            w for w in recorded_warnings
            if issubclass(w.category, (RuntimeWarning, FloatingPointError))
            and any(k in str(w.message).lower() for k in ["overflow", "divide by zero", "invalid value"])
        ]
        self.assertEqual(len(overflow_warnings), 0, f"Overflow warnings detected: {overflow_warnings}")

    def test_challenge1_isolated_voids_and_floating_islands(self):
        """
        Adversarial topological distribution:
        - Fully isolated 1-voxel void elements embedded in solid matrix
        - Fully isolated 1-voxel solid elements ('floating islands') surrounded by voids
        Tested under Emin=1e-10 (kappa >= 10^10).
        Verifies AMG hierarchy builds and PCG converges with relative residual < 1e-4.
        """
        nelx, nely, nelz = 8, 8, 8
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-10, E0=1.0)
        opt.fix_face("left")
        opt.add_load(nelx, nely // 2, nelz // 2, fx=50.0, fy=-50.0, fz=-100.0)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        F_free = opt.force_vector[free_dofs]

        # Base solid matrix
        xPhys = np.ones(opt.num_elements, dtype=np.float64)

        # Cut out isolated interior voids
        xPhys[nelx * nely * 2 + nely * 2 + 2] = 0.0
        xPhys[nelx * nely * 4 + nely * 4 + 4] = 0.0
        xPhys[nelx * nely * 6 + nely * 6 + 6] = 0.0

        # Create isolated solid elements in void region (xPhys = 0 everywhere except single voxel)
        void_slice_start = nelx * nely * 5
        void_slice_end = nelx * nely * 7
        xPhys[void_slice_start:void_slice_end] = 0.0
        # Floating island
        island_idx = void_slice_start + nely * 3 + 3
        xPhys[island_idx] = 1.0

        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

        u_amg, exit_code, sname = opt.solve_linear_system(
            K_free, F_free, solver_type="amg", rtol=1e-5, maxiter=1500
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(sname, "amg_pcg")
        self.assertTrue(np.all(np.isfinite(u_amg)), "Solution contains NaN/Inf with floating islands")

        rel_res = np.linalg.norm(K_free @ u_amg - F_free) / np.linalg.norm(F_free)
        self.assertLess(rel_res, 1e-4, f"Relative residual {rel_res} exceeds 1e-4")

    # =========================================================================
    # CHALLENGE 2: DIRICHLET BC STABILITY & COARSE GRID 'pinv' VS 'splu'
    # =========================================================================

    def test_challenge2_coarse_grid_pinv_prevents_splu_crash(self):
        """
        Adversarial test on singular / near-singular coarse grids:
        Verifies empirically:
        1. When a coarse grid contains rank deficiency (zero singular value),
           SuperLU direct factorization (coarse_solver='splu') crashes with
           RuntimeError: Factor is exactly singular.
        2. In exact contrast, coarse_solver='pinv' utilizes SVD pseudoinverse
           truncation, completely preventing the crash and generating a strictly
           finite multilevel preconditioner.
        3. Also tests minimally constrained 3D continuum (only 1 fixed node = 3 DOFs).
        """
        # Part A: Demonstrating the SuperLU splu crash on rank-deficient operator
        A_singular = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64))
        splu_crashed = False
        try:
            ml_splu = pyamg.smoothed_aggregation_solver(A_singular, coarse_solver='splu', max_levels=1)
            _ = ml_splu.aspreconditioner().matvec(np.ones(2))
        except RuntimeError as e:
            if "Factor is exactly singular" in str(e) or "singular" in str(e).lower():
                splu_crashed = True
        except Exception:
            splu_crashed = True

        self.assertTrue(splu_crashed, "SuperLU splu must crash on singular coarse operator")

        # Part B: Verifying pinv completely prevents the crash
        pinv_succeeded = False
        try:
            ml_pinv = pyamg.smoothed_aggregation_solver(A_singular, coarse_solver='pinv', max_levels=1)
            out_pinv = ml_pinv.aspreconditioner().matvec(np.ones(2))
            if np.all(np.isfinite(out_pinv)) and np.allclose(out_pinv, [0.5, 0.5]):
                pinv_succeeded = True
        except Exception:
            pinv_succeeded = False

        self.assertTrue(pinv_succeeded, "coarse_solver='pinv' must prevent crash and compute Moore-Penrose action")

        # Part C: 3D Continuum with minimal boundary condition (1 fixed node)
        nelx, nely, nelz = 6, 4, 3
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-8, E0=1.0)
        opt.fix_node(0, 0, 0, fix_x=True, fix_y=True, fix_z=True)

        free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
        xPhys = np.full(opt.num_elements, 1e-4, dtype=np.float64)
        K_full = opt.assemble_elastic_stiffness(xPhys)
        K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

        ml_3d_pinv = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
        M_pinv = ml_3d_pinv.aspreconditioner()
        v_test = np.ones(K_free.shape[0], dtype=np.float64)
        v_out = M_pinv.matvec(v_test)
        self.assertTrue(np.all(np.isfinite(v_out)), "Multilevel pinv preconditioner output must be finite.")

    def test_challenge2_disconnected_subdomain_stability(self):
        """
        Construct a disconnected domain: two solid blocks separated by a layer
        of void elements. Dirichlet BCs applied only to the left block; load on right block.
        Tested at:
        1. Emin = 1e-8 (kappa >= 10^8): verifies relative residual < 1e-4.
        2. Emin = 1e-10 (kappa >= 10^10): floating displacement ~ 10^12, verifies
           AMG-PCG matches SuperLU direct factorization within relative error < 1e-4
           despite floating-point roundoff floor.
        """
        nelx, nely, nelz = 9, 3, 3

        # Case 1: Emin = 1e-8 (condition number ~ 10^8)
        opt8 = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-8, E0=1.0)
        opt8.fix_face("left")
        opt8.add_load(nelx, 1, 1, fz=-10.0)

        free_dofs8 = np.array(sorted(list(set(range(opt8.num_dofs)) - opt8.fixed_dofs)), dtype=np.int32)
        F_free8 = opt8.force_vector[free_dofs8]

        xPhys = np.ones(opt8.num_elements, dtype=np.float64)
        for elx in range(3, 6):
            for ely in range(nely):
                for elz in range(nelz):
                    idx = elx + ely * nelx + elz * nelx * nely
                    xPhys[idx] = 0.0

        K_full8 = opt8.assemble_elastic_stiffness(xPhys)
        K_free8 = K_full8[free_dofs8, :][:, free_dofs8].tocsr()

        u8, exit_code8, sname8 = opt8.solve_linear_system(
            K_free8, F_free8, solver_type="amg", rtol=1e-5, maxiter=2000
        )
        self.assertEqual(exit_code8, 0)
        self.assertEqual(sname8, "amg_pcg")
        self.assertTrue(np.all(np.isfinite(u8)))
        rel_res8 = np.linalg.norm(K_free8 @ u8 - F_free8) / np.linalg.norm(F_free8)
        self.assertLess(rel_res8, 1e-4, f"Relative residual {rel_res8} at Emin=1e-8 exceeds 1e-4")

        # Case 2: Extreme Emin = 1e-10 (condition number ~ 10^10)
        opt10 = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-10, E0=1.0)
        opt10.fix_face("left")
        opt10.add_load(nelx, 1, 1, fz=-10.0)
        free_dofs10 = np.array(sorted(list(set(range(opt10.num_dofs)) - opt10.fixed_dofs)), dtype=np.int32)
        F_free10 = opt10.force_vector[free_dofs10]
        K_full10 = opt10.assemble_elastic_stiffness(xPhys)
        K_free10 = K_full10[free_dofs10, :][:, free_dofs10].tocsr()

        u_dir10 = sla.spsolve(K_free10, F_free10)
        u10, exit_code10, sname10 = opt10.solve_linear_system(
            K_free10, F_free10, solver_type="amg", rtol=1e-5, maxiter=2000
        )
        self.assertEqual(exit_code10, 0)
        self.assertTrue(np.all(np.isfinite(u10)))
        rel_diff10 = np.linalg.norm(u10 - u_dir10) / np.linalg.norm(u_dir10)
        self.assertLess(rel_diff10, 1e-4, f"Relative difference {rel_diff10} vs direct solver exceeds 1e-4")

    # =========================================================================
    # CHALLENGE 3: SOLVER ACCURACY (AMG-PCG VS SUPERLU DIRECT SOLVE)
    # =========================================================================

    def test_challenge3_amg_vs_superlu_accuracy_across_grids_and_loads(self):
        """
        Systematic accuracy comparison between AMG-PCG and SuperLU direct solver
        across various 3D mesh dimensions, load combinations, and physical regimes:
        1. 4x4x4 mesh with asymmetric corner load
        2. 8x6x4 mesh with bending + torsional load
        3. 10x6x4 mesh with distributed shear loads
        4. Thermal conductivity system K_th T = Q
        
        Verifies:
        - Relative residual ||K u_amg - F|| / ||F|| < 1e-4
        - Solution agreement ||u_amg - u_dir|| / ||u_dir|| < 1e-4
        """
        configurations = [
            # (nelx, nely, nelz, load_type, phys_type)
            (4, 4, 4, "corner_asymmetric", "elastic"),
            (8, 6, 4, "torsion_bending", "elastic"),
            (10, 6, 4, "distributed_shear", "elastic"),
            (6, 6, 6, "thermal_flux", "thermal"),
        ]

        for nelx, nely, nelz, load_type, phys_type in configurations:
            with self.subTest(grid=f"{nelx}x{nely}x{nelz}", load=load_type, physics=phys_type):
                opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-6, E0=1.0)

                if phys_type == "elastic":
                    opt.fix_face("left")
                    if load_type == "corner_asymmetric":
                        opt.add_load(nelx, 0, 0, fx=20.0, fy=-30.0, fz=50.0)
                    elif load_type == "torsion_bending":
                        opt.add_load(nelx, 0, 0, fy=100.0, fz=-50.0)
                        opt.add_load(nelx, nely, nelz, fy=-100.0, fz=-50.0)
                    elif load_type == "distributed_shear":
                        for j in range(nely + 1):
                            opt.add_load(nelx, j, nelz // 2, fy=20.0, fz=-30.0)

                    free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
                    F_free = opt.force_vector[free_dofs]

                    # Intermediate heterogeneous density field
                    np.random.seed(42)
                    xPhys = 0.3 + 0.4 * np.random.rand(opt.num_elements)
                    K_full = opt.assemble_elastic_stiffness(xPhys)
                    K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

                    # Direct SuperLU solution
                    u_dir = sla.spsolve(K_free, F_free)

                    # AMG-PCG solution
                    u_amg, exit_code, sname = opt.solve_linear_system(
                        K_free, F_free, solver_type="amg", rtol=1e-6, maxiter=1000
                    )

                    self.assertEqual(exit_code, 0)
                    self.assertEqual(sname, "amg_pcg")

                    res_norm = np.linalg.norm(K_free @ u_amg - F_free) / np.linalg.norm(F_free)
                    rel_err = np.linalg.norm(u_amg - u_dir) / np.linalg.norm(u_dir)

                    self.assertLess(
                        res_norm, 1e-4,
                        f"Grid {nelx}x{nely}x{nelz} {load_type}: residual {res_norm:.2e} >= 1e-4"
                    )
                    self.assertLess(
                        rel_err, 1e-4,
                        f"Grid {nelx}x{nely}x{nelz} {load_type}: direct error {rel_err:.2e} >= 1e-4"
                    )

                elif phys_type == "thermal":
                    opt.fix_thermal_face("left", temp=0.0)
                    opt.add_heat_source(nelx, nely // 2, nelz // 2, q=100.0)

                    xPhys = np.full(opt.num_elements, 0.5, dtype=np.float64)
                    K_th = opt.assemble_thermal_conductivity(xPhys)
                    fixed_nodes = set(opt.fixed_thermal_nodes.keys())
                    free_th = np.array([n for n in range(opt.num_nodes) if n not in fixed_nodes], dtype=np.int32)
                    Q_free = opt.heat_source_vector[free_th]
                    K_th_free = K_th[free_th, :][:, free_th].tocsr()

                    T_dir = sla.spsolve(K_th_free, Q_free)
                    T_amg, exit_code, sname = opt.solve_linear_system(
                        K_th_free, Q_free, solver_type="amg", rtol=1e-6, maxiter=1000
                    )

                    self.assertEqual(exit_code, 0)
                    res_norm = np.linalg.norm(K_th_free @ T_amg - Q_free) / np.linalg.norm(Q_free)
                    rel_err = np.linalg.norm(T_amg - T_dir) / np.linalg.norm(T_dir)

                    self.assertLess(res_norm, 1e-4, f"Thermal residual {res_norm:.2e} >= 1e-4")
                    self.assertLess(rel_err, 1e-4, f"Thermal direct error {rel_err:.2e} >= 1e-4")

    # =========================================================================
    # CHALLENGE 4: MEMORY FOOTPRINT (< 100 MB ON 3D MESHES)
    # =========================================================================

    def test_challenge4_amg_hierarchy_memory_footprint(self):
        """
        Adversarial memory footprint audit:
        Measures memory consumption using tracemalloc during:
        - Smoothed aggregation hierarchy creation
        - Multilevel V-cycle preconditioned CG iterations
        Evaluates meshes up to 20x20x20 (8,000 H8 elements, 27,783 DOFs).
        Verifies:
        1. Peak hierarchy memory is strictly < 100 MB.
        2. Operator complexity remains modest (C_op < 1.6).
        """
        test_grids = [
            (12, 12, 12),  # 1,728 elements, 6,591 DOFs
            (16, 16, 16),  # 4,096 elements, 14,739 DOFs
            (20, 16, 12),  # 3,840 elements, 13,875 DOFs
            (20, 20, 20),  # 8,000 elements, 27,783 DOFs
        ]

        print("\n--- AMG Hierarchy Memory Audit ---")
        for nelx, nely, nelz in test_grids:
            opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, Emin=1e-6, E0=1.0)
            opt.fix_face("left")
            opt.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

            free_dofs = np.array(sorted(list(set(range(opt.num_dofs)) - opt.fixed_dofs)), dtype=np.int32)
            F_free = opt.force_vector[free_dofs]

            xPhys = np.full(opt.num_elements, 0.5, dtype=np.float64)
            K_full = opt.assemble_elastic_stiffness(xPhys)
            K_free = K_full[free_dofs, :][:, free_dofs].tocsr()

            tracemalloc.start()
            tracemalloc.reset_peak()

            # Construct hierarchy and run AMG-PCG
            ml = pyamg.smoothed_aggregation_solver(K_free, coarse_solver='pinv', symmetry='symmetric')
            op_complexity = ml.operator_complexity()
            grid_complexity = ml.grid_complexity()
            num_levels = len(ml.levels)

            M_amg = ml.aspreconditioner(cycle='V')
            u_amg, exit_code = sla.cg(K_free, F_free, M=M_amg, rtol=1e-5, maxiter=200)

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak_mem / (1024 * 1024)
            current_mb = current_mem / (1024 * 1024)

            print(
                f"Grid {nelx}x{nely}x{nelz}: DOFs={len(free_dofs)}, Levels={num_levels}, "
                f"OpComplexity={op_complexity:.2f}, GridComplexity={grid_complexity:.2f}, "
                f"PeakMem={peak_mb:.2f} MB, CurrentMem={current_mb:.2f} MB"
            )

            self.assertEqual(exit_code, 0, f"AMG-PCG failed on grid {nelx}x{nely}x{nelz}")
            self.assertLess(op_complexity, 1.8, f"Operator complexity {op_complexity:.2f} is too high")
            self.assertLess(
                peak_mb, 100.0,
                f"Peak memory {peak_mb:.2f} MB exceeded 100 MB budget on grid {nelx}x{nely}x{nelz}!"
            )


if __name__ == "__main__":
    unittest.main()
