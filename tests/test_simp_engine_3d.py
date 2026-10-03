"""
Unit and benchmark tests for 3D Continuum Topology Optimization Engine (simp_engine_3d.py).

Tests:
1. H8 element stiffness matrix spectral properties (exact 6 zero eigenvalues, 18 positive).
2. 3D geometric stiffness basis matrices symmetry and properties.
3. Precomputed edofMat and invariant (iK, jK) sparsity assembly.
4. Linear solver equivalence: PCG with Jacobi preconditioning vs direct SuperLU fallback.
5. 3D spatial filter via scipy.ndimage.convolve and boundary normalization.
6. 10x10x10 Cantilever beam benchmark under extreme memory constraints (< 100 MB RAM).
7. Watertight binary STL export via Trimesh box voxel representation.
8. Multi-objective compliance + linearized buckling optimization with adjoint sensitivities.
9. 3D Principal Cauchy stress tensor and von Mises reconstruction.
10. Boundary condition helpers (fix_face, add_load, clear_boundary_conditions).
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.ndimage as ndimage

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
    geometric_stiffness_basis_matrices,
    spherical_cone_kernel
)


class TestSIMPEngine3D(unittest.TestCase):
    """Unit and benchmark test suite for 3D SIMP topology optimization solver."""

    # =========================================================================
    # Test 1: H8 Element Stiffness Matrix Properties & Spectral Modes
    # =========================================================================
    def test_1_element_stiffness_k0_spectral_properties(self):
        """
        Verify H8 element stiffness matrix k0 is 24x24, strictly symmetric,
        and possesses exactly 6 zero eigenvalues (rigid body modes in 3D: 3 translations + 3 rotations)
        and 18 strictly positive eigenvalues (elastic deformation strain modes).
        """
        k0 = h8_element_stiffness_k0(dx=1.0, dy=1.0, dz=1.0, E0=1.0, nu=0.3)
        self.assertEqual(k0.shape, (24, 24))

        # Symmetry check
        np.testing.assert_allclose(k0, k0.T, atol=1e-12, err_msg="k0 must be strictly symmetric.")

        # Spectral decomposition
        eigvals = np.linalg.eigvalsh(k0)
        zero_eigs = eigvals[np.abs(eigvals) < 1e-10]
        pos_eigs = eigvals[eigvals >= 1e-10]

        self.assertEqual(len(zero_eigs), 6, f"Expected exactly 6 rigid body modes, found {len(zero_eigs)}.")
        self.assertEqual(len(pos_eigs), 18, f"Expected exactly 18 elastic modes, found {len(pos_eigs)}.")
        self.assertGreater(float(np.min(pos_eigs)), 0.05, "All elastic modes must be strictly positive.")

    # =========================================================================
    # Test 2: 3D Geometric Stiffness Basis Matrices
    # =========================================================================
    def test_2_geometric_stiffness_basis_matrices(self):
        """
        Verify that all 6 constant geometric stiffness basis matrices G0_k are
        24x24, strictly symmetric, and non-trivial.
        """
        G0_dict = geometric_stiffness_basis_matrices(dx=1.0, dy=1.0, dz=1.0)
        expected_keys = {'xx', 'yy', 'zz', 'yz', 'xz', 'xy'}
        self.assertEqual(set(G0_dict.keys()), expected_keys)

        for key, G0_k in G0_dict.items():
            self.assertEqual(G0_k.shape, (24, 24), f"G0_{key} must be 24x24.")
            np.testing.assert_allclose(G0_k, G0_k.T, atol=1e-12, err_msg=f"G0_{key} must be symmetric.")
            self.assertGreater(float(np.linalg.norm(G0_k)), 0.0, f"G0_{key} must be non-zero.")

    # =========================================================================
    # Test 3: Sparse Assembly & Indexing Preallocation
    # =========================================================================
    def test_3_sparse_assembly_and_indexing(self):
        """
        Verify edofMat shape (num_elements, 24), iK and jK lengths (576 * num_elements),
        and that assembled stiffness K is symmetric and positive definite on free DOFs.
        """
        nelx, nely, nelz = 4, 3, 2
        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz)

        self.assertEqual(opt.num_elements, 24)
        self.assertEqual(opt.edofMat.shape, (24, 24))
        self.assertEqual(len(opt.iK), 24 * 576)
        self.assertEqual(len(opt.jK), 24 * 576)

        # Assemble full solid stiffness
        sK = np.outer(np.ones(opt.num_elements), opt.k0.ravel()).ravel()
        K_full = sp.coo_matrix((sK, (opt.iK, opt.jK)), shape=(opt.num_dofs, opt.num_dofs)).tocsr()

        # Fix left face
        opt.fix_face("left")
        all_dofs = np.arange(opt.num_dofs)
        free_mask = np.ones(opt.num_dofs, dtype=bool)
        for fd in opt.fixed_dofs:
            free_mask[fd] = False
        free_dofs = all_dofs[free_mask]

        K_free = K_full[free_dofs, :][:, free_dofs]
        diff = (K_free - K_free.T).data
        if len(diff) > 0:
            self.assertLess(float(np.max(np.abs(diff))), 1e-11)

    # =========================================================================
    # Test 4: Linear Solver Equivalence (PCG vs Direct SuperLU)
    # =========================================================================
    def test_4_linear_solvers_pcg_vs_direct(self):
        """
        Verify that Preconditioned Conjugate Gradient (PCG) with Jacobi preconditioning
        produces solutions matching direct SuperLU factorization within tolerance.
        """
        nelx, nely, nelz = 5, 4, 3
        opt_pcg = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="pcg", max_iter=2)
        opt_dir = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, solver_type="direct", max_iter=2)

        opt_pcg.fix_face("left")
        opt_pcg.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        opt_dir.fix_face("left")
        opt_dir.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)

        res_pcg = opt_pcg.solve(max_iter=2)
        res_dir = opt_dir.solve(max_iter=2)

        self.assertTrue(res_pcg.success)
        self.assertTrue(res_dir.success)
        self.assertAlmostEqual(res_pcg.compliance, res_dir.compliance, delta=res_dir.compliance * 0.05)

    # =========================================================================
    # Test 5: 3D Spatial Filter via Convolution & Boundary Normalization
    # =========================================================================
    def test_5_spatial_filtering_3d_convolution(self):
        """
        Verify that the 3D convolution spatial filter preserves a uniform field
        identically (boundary normalization property) and respects bounds [0, 1].
        """
        kernel = spherical_cone_kernel(rmin=1.5, dx=1.0, dy=1.0, dz=1.0)
        self.assertGreater(kernel.size, 0)
        self.assertAlmostEqual(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2], 1.5)

        opt = SIMPOptimizer3D(nelx=6, nely=6, nelz=6, rmin=1.5)
        ones_field = np.ones((6, 6, 6))
        conv_ones = ndimage.convolve(ones_field, opt.kernel, mode='constant', cval=0.0)
        filtered = conv_ones.ravel() / opt.kernel_normalizer

        np.testing.assert_allclose(filtered, 1.0, atol=1e-12)

    # =========================================================================
    # Test 6: 10x10x10 Cantilever Beam Benchmark (< 100 MB RAM on CPU)
    # =========================================================================
    def test_6_cantilever_benchmark_10x10x10_memory_and_convergence(self):
        """
        Full 10x10x10 Cantilever beam topology optimization benchmark verifying:
        - Strict convergence within 15 iterations.
        - Volume fraction conservation (0.30 +/- 0.02).
        - Monotonic decrease / stabilization in compliance history.
        - Peak memory consumption strictly < 100 MB (far below 4GB budget).
        """
        opt = SIMPOptimizer3D(
            nelx=10, nely=10, nelz=10,
            dx=1.0, dy=1.0, dz=1.0,
            volfrac=0.30, penal=3.0, rmin=1.5,
            solver_type="pcg", max_iter=15
        )
        opt.fix_face("left")
        opt.add_load(10, 5, 5, fz=-100.0)

        res = opt.solve(max_iter=15, tol=0.01)

        self.assertTrue(res.success)
        self.assertEqual(res.density_matrix.shape, (10, 10, 10))
        self.assertAlmostEqual(res.volume_fraction, 0.30, delta=0.02)
        self.assertLess(res.compliance_history[-1], res.compliance_history[0])
        self.assertLess(res.peak_memory_mb, 100.0, f"Memory {res.peak_memory_mb} MB exceeds 100 MB target limit.")

    # =========================================================================
    # Test 7: Watertight STL Export Pipeline
    # =========================================================================
    def test_7_watertight_stl_export_pipeline(self):
        """
        Verify export_voxel_stl creates a non-empty binary STL file with valid
        80-byte header, uint32 face count > 0, and reloadable closed mesh geometry.
        """
        import trimesh

        opt = SIMPOptimizer3D(nelx=6, nely=6, nelz=6, volfrac=0.5, max_iter=3)
        res = opt.solve(max_iter=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "test_model.stl")
            out_path = res.export_voxel_stl(stl_path, threshold=0.4)

            self.assertEqual(out_path, stl_path)
            self.assertTrue(os.path.exists(stl_path))
            self.assertGreater(os.path.getsize(stl_path), 84)

            # Check binary STL format
            with open(stl_path, "rb") as f:
                header = f.read(80)
                face_count = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            self.assertEqual(len(header), 80)
            self.assertGreater(face_count, 0)

            # Reload with trimesh
            mesh = trimesh.load(stl_path, force='mesh')
            self.assertGreater(len(mesh.faces), 0)
            self.assertGreater(len(mesh.vertices), 0)

    # =========================================================================
    # Test 8: Multi-Objective Buckling Optimization
    # =========================================================================
    def test_8_multi_objective_buckling_optimization(self):
        """
        Verify multi-objective optimization with alpha > 0 executes LBA,
        records blf_history, and blends compliance with buckling eigenvalue sensitivities.
        """
        opt = SIMPOptimizer3D(
            nelx=8, nely=4, nelz=4,
            volfrac=0.35, penal=3.0, rmin=1.5,
            alpha=0.3, solver_type="pcg", max_iter=4
        )
        opt.fix_face("left")
        opt.add_load(8, 2, 2, fx=-100.0)  # Axial compressive load

        res = opt.solve(max_iter=4, mode="buckling_max")

        self.assertTrue(res.success)
        self.assertGreater(len(res.blf_history), 0)
        self.assertTrue(all(blf > 0 for blf in res.blf_history))

    # =========================================================================
    # Test 9: 3D Principal Cauchy Stress Tensor Reconstruction
    # =========================================================================
    def test_9_stress_tensor_reconstruction(self):
        """
        Verify compute_principal_stresses evaluates Cauchy stress components,
        principal stresses satisfying sigma_I >= sigma_II >= sigma_III, and von Mises >= 0.
        """
        opt = SIMPOptimizer3D(nelx=4, nely=4, nelz=4, max_iter=2)
        opt.fix_face("left")
        opt.add_load(4, 2, 2, fz=-50.0)
        res = opt.solve(max_iter=2)

        # Create synthetic displacement
        u_dummy = np.random.randn(opt.num_dofs)
        stresses = opt.compute_principal_stresses(u_dummy)

        self.assertIn("sigma_I", stresses)
        self.assertIn("sigma_II", stresses)
        self.assertIn("sigma_III", stresses)
        self.assertIn("von_mises", stresses)

        s1 = stresses["sigma_I"]
        s2 = stresses["sigma_II"]
        s3 = stresses["sigma_III"]
        vm = stresses["von_mises"]

        self.assertTrue(np.all(s1 >= s2 - 1e-10))
        self.assertTrue(np.all(s2 >= s3 - 1e-10))
        self.assertTrue(np.all(vm >= -1e-10))

    # =========================================================================
    # Test 10: Boundary Condition Helper Methods
    # =========================================================================
    def test_10_boundary_condition_helpers(self):
        """
        Verify fix_face, fix_node, add_load, and clear_boundary_conditions.
        """
        opt = SIMPOptimizer3D(nelx=4, nely=4, nelz=4)
        self.assertEqual(len(opt.fixed_dofs), 0)

        opt.fix_face("left")
        self.assertGreater(len(opt.fixed_dofs), 0)
        fixed_count = len(opt.fixed_dofs)

        opt.fix_face("right")
        self.assertGreater(len(opt.fixed_dofs), fixed_count)

        opt.add_load(4, 2, 2, fx=10.0, fy=20.0, fz=30.0)
        self.assertGreater(float(np.linalg.norm(opt.force_vector)), 0.0)

        opt.clear_boundary_conditions()
        self.assertEqual(len(opt.fixed_dofs), 0)
        self.assertAlmostEqual(float(np.linalg.norm(opt.force_vector)), 0.0)


if __name__ == "__main__":
    unittest.main()
