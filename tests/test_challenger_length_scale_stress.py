"""
test_challenger_length_scale_stress.py
Independent Adversarial Stress Suite by Challenger (challenger_length_scale_stress).

Scope:
1. Minimum length-scale feature control in 'thermo_elastic' mode:
   - Full optimization comparison: rmin >= 2.0 * max(dx, dy, dz) vs unconstrained (rmin = 0.9).
   - 3D morphological binary erosion using scipy.ndimage.binary_erosion with 2x2x2 structuring element.
   - Quantitative measurement of 1-voxel thin branch fraction:
     thin_fraction = (solid_voxels - eroded_solid_voxels) / solid_voxels.
   - Rigorous assertion that constrained run demonstrably suppresses 1-voxel thin branches.
   - Anisotropic grid length-scale test under non-uniform voxel dimensions (dx != dy != dz).
2. Opposing/conflicting loads in 'thermo_elastic' mode:
   - Strong mechanical shear/tensile loads opposing thermal expansion (Delta T > 200 K).
   - Severe thermal contraction (Delta T = -250 K) opposing outward mechanical tension.
   - Low volume fraction (Vf = 0.15) under severe thermal gradient (Delta T = 250 K).
   - Verification of smooth MMA convergence without NaN/Inf, divergence, or asymptote collapse.
   - Verification of physical invariants: finite displacements, bounded densities, volume conservation.
"""

import os
import sys
import unittest
import numpy as np
import scipy.sparse as sp
import scipy.ndimage as ndimage
from scipy.ndimage import binary_erosion

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_3d import SIMPOptimizer3D, SIMPResult3D


def compute_3d_thin_fraction(density_matrix: np.ndarray, threshold: float = 0.35) -> dict:
    """
    Evaluates 3D morphological feature characteristics using a 2x2x2 structuring element.
    
    A 1-voxel thin branch cannot contain a 2x2x2 cube of solid voxels.
    Therefore, binary erosion with structure=np.ones((2, 2, 2)) strips away all
    features of thickness 1 voxel.
    
    Returns:
        dict with:
            solid_count: int
            eroded_count: int
            thin_count: int
            thin_fraction: float (thin_count / solid_count)
            retained_core_fraction: float (eroded_count / solid_count)
    """
    solid_mask = (density_matrix >= threshold)
    solid_count = int(np.sum(solid_mask))
    if solid_count == 0:
        return {
            "solid_count": 0,
            "eroded_count": 0,
            "thin_count": 0,
            "thin_fraction": 0.0,
            "retained_core_fraction": 0.0,
        }
    
    # 2x2x2 structuring element evaluates 3D feature thickness < 2 voxels
    struct_2x2x2 = np.ones((2, 2, 2), dtype=bool)
    eroded_mask = binary_erosion(solid_mask, structure=struct_2x2x2)
    eroded_count = int(np.sum(eroded_mask))
    
    thin_voxels_mask = solid_mask & (~eroded_mask)
    thin_count = int(np.sum(thin_voxels_mask))
    
    thin_fraction = float(thin_count / solid_count)
    retained_core_fraction = float(eroded_count / solid_count)
    
    return {
        "solid_count": solid_count,
        "eroded_count": eroded_count,
        "thin_count": thin_count,
        "thin_fraction": thin_fraction,
        "retained_core_fraction": retained_core_fraction,
    }


class TestChallengerLengthScaleAndOpposingStress(unittest.TestCase):
    """Adversarial stress tests for length scale feature control and conflicting loads."""

    def test_stress1_thermo_elastic_length_scale_suppression(self):
        """
        Adversarial Test 1: Minimum Length-Scale Suppression in 'thermo_elastic' Mode.
        
        Compares full topology optimization in 'thermo_elastic' mode under:
          Run A: Unconstrained (rmin = 0.9 * max(dx, dy, dz) = 0.9)
          Run B: Constrained (rmin = 2.0 * max(dx, dy, dz) = 2.0)
        
        Quantifies 1-voxel thin branches via 3D morphological erosion (2x2x2 structuring element).
        Asserts that the constrained run demonstrably suppresses thin 1-voxel branches
        and increases retained bulk core volume fraction.
        """
        nelx, nely, nelz = 10, 10, 6
        dx, dy, dz = 1.0, 1.0, 1.0
        volfrac = 0.35
        max_dim = max(dx, dy, dz)

        # -------------------------------------------------------------
        # Run A: Unconstrained filter (rmin = 0.9)
        # -------------------------------------------------------------
        rmin_unconstrained = 0.9 * max_dim
        opt_un = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            dx=dx, dy=dy, dz=dz,
            volfrac=volfrac, penal=3.0,
            rmin=rmin_unconstrained,
            solver_type="direct", T_ref=0.0
        )
        opt_un.fix_face("left")
        # Mechanical load: transverse tip shear/downward load
        opt_un.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        # Thermal load: heat source at loaded tip and cold sink at fixed base
        opt_un.fix_thermal_face("left", temp=0.0)
        opt_un.fix_thermal_face("right", temp=80.0)

        res_un = opt_un.solve(max_iter=15, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")
        self.assertTrue(res_un.success, "Unconstrained optimization must complete.")

        metrics_un = compute_3d_thin_fraction(res_un.density_matrix, threshold=0.35)

        # -------------------------------------------------------------
        # Run B: Constrained filter (rmin = 2.0 * max(dx, dy, dz))
        # -------------------------------------------------------------
        rmin_constrained = 2.0 * max_dim
        opt_con = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            dx=dx, dy=dy, dz=dz,
            volfrac=volfrac, penal=3.0,
            rmin=rmin_constrained,
            solver_type="direct", T_ref=0.0
        )
        opt_con.fix_face("left")
        opt_con.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        opt_con.fix_thermal_face("left", temp=0.0)
        opt_con.fix_thermal_face("right", temp=80.0)

        res_con = opt_con.solve(max_iter=15, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")
        self.assertTrue(res_con.success, "Constrained optimization must complete.")

        metrics_con = compute_3d_thin_fraction(res_con.density_matrix, threshold=0.35)

        print("\n--- TEST 1: LENGTH SCALE MORPHOLOGICAL EROSION COMPARISON ---")
        print(f"Unconstrained (rmin={rmin_unconstrained:.1f}): Solid={metrics_un['solid_count']}, "
              f"Eroded={metrics_un['eroded_count']}, Thin Fraction={metrics_un['thin_fraction']:.2%}, "
              f"Retained Core={metrics_un['retained_core_fraction']:.2%}")
        print(f"Constrained   (rmin={rmin_constrained:.1f}): Solid={metrics_con['solid_count']}, "
              f"Eroded={metrics_con['eroded_count']}, Thin Fraction={metrics_con['thin_fraction']:.2%}, "
              f"Retained Core={metrics_con['retained_core_fraction']:.2%}")

        # Verification Criteria:
        # 1. Both runs produced solid material conserving target volume fraction
        self.assertGreater(metrics_un["solid_count"], 0, "Unconstrained run must have solid elements.")
        self.assertGreater(metrics_con["solid_count"], 0, "Constrained run must have solid elements.")
        self.assertAlmostEqual(res_un.volume_fraction, volfrac, delta=0.03)
        self.assertAlmostEqual(res_con.volume_fraction, volfrac, delta=0.03)

        # 2. Constrained run must demonstrably suppress 1-voxel thin branches:
        #    - Thin branch fraction must be lower in constrained run
        #    - Retained core fraction must be higher in constrained run
        self.assertLess(
            metrics_con["thin_fraction"], metrics_un["thin_fraction"],
            f"Constrained run thin fraction ({metrics_con['thin_fraction']:.2%}) was not less than "
            f"unconstrained thin fraction ({metrics_un['thin_fraction']:.2%})."
        )
        self.assertGreater(
            metrics_con["retained_core_fraction"], metrics_un["retained_core_fraction"],
            f"Constrained run retained core ({metrics_con['retained_core_fraction']:.2%}) was not greater than "
            f"unconstrained retained core ({metrics_un['retained_core_fraction']:.2%})."
        )

    def test_stress2_opposing_conflicting_thermo_elastic_loads(self):
        """
        Adversarial Test 2: Opposing/Conflicting Loads with Delta T > 200 K.
        
        Stress-tests MMA numerical stability under conflicting physical driving forces:
          1. Severe mechanical tension & shear pulling in +x and +y directions:
             Fx = +300.0, Fy = +150.0
          2. Severe opposing thermal expansion: Delta T = 250.0 K (> 200 K threshold)
             Fixed cold face at left (T = 0 K), high temperature face at right (T = 250 K).
             Thermal expansion generates large compressive reaction forces against the fixed wall
             that directly fight the mechanical tension.
        
        Verifies:
          - Zero NaN or Inf values across all optimization iterations.
          - Monotonic or stable objective convergence without asymptote breakdown.
          - Volume conservation within 2% of target.
          - Displacements and temperatures remain physically bounded.
        """
        nelx, nely, nelz = 8, 8, 4
        volfrac = 0.30
        delta_T = 250.0  # Delta T > 200 K

        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, penal=3.0, rmin=1.8,
            solver_type="amg", T_ref=0.0, alpha_th=1.0e-5
        )
        opt.fix_face("left")

        # Strong mechanical tensile and shear forces at tip
        opt.add_load(nelx, nely // 2, nelz // 2, fx=300.0, fy=150.0, fz=-100.0)

        # Severe opposing thermal condition: Delta T = 250 K
        opt.fix_thermal_face("left", temp=0.0)
        opt.fix_thermal_face("right", temp=delta_T)

        res = opt.solve(max_iter=15, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")

        print("\n--- TEST 2: OPPOSING CONFLICTING LOADS (Delta T = 250 K) ---")
        print(f"Iterations: {res.iterations_run}")
        print(f"Final Compliance: {res.compliance:.4e}")
        print(f"Final Volume Fraction: {res.volume_fraction:.4f} (target: {volfrac:.2f})")
        print(f"Max Displacement Norm: {float(np.max(np.abs(res.displacements))):.4e}")
        print(f"Max Temperature: {float(np.max(res.temperatures)):.2f} K")

        # 1. Verification of clean execution and no NaNs
        self.assertTrue(res.success, "MMA must converge or finish cleanly under opposing loads.")
        self.assertTrue(np.all(np.isfinite(res.compliance_history)), "Compliance history contains NaN or Inf.")
        self.assertTrue(np.all(np.isfinite(res.displacements)), "Displacements contain NaN or Inf.")
        self.assertTrue(np.all(np.isfinite(res.temperatures)), "Temperatures contain NaN or Inf.")
        self.assertTrue(np.all(np.isfinite(res.density_matrix)), "Density matrix contains NaN or Inf.")

        # 2. Temperature boundary conditions verified
        self.assertGreaterEqual(float(np.max(res.temperatures)), delta_T * 0.95,
                                f"Max temperature must reach ~{delta_T} K.")

        # 3. Density boundedness and volume conservation
        self.assertTrue(np.all(res.density_matrix >= 0.0) and np.all(res.density_matrix <= 1.0),
                        "Densities must remain strictly in [0, 1].")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.02,
                               msg=f"Volume fraction {res.volume_fraction:.4f} violated target {volfrac}.")

        # 4. Smooth convergence: compliance does not blow up or oscillate wildly
        c_hist = res.compliance_history
        self.assertGreater(len(c_hist), 2, "Must execute multiple iterations.")
        self.assertLess(c_hist[-1], c_hist[0] * 5.0,
                        "Compliance diverged under conflicting thermo-elastic MMA.")

    def test_stress3_extreme_shear_and_high_thermal_gradient_300K(self):
        """
        Adversarial Test 3: Extreme Pure Shear Load with Transverse Delta T = 300 K.
        
        Shear load fy=+200.0 applied along the top edge, while bottom face is fixed mechanically.
        Transverse thermal gradient: bottom fixed at T = 0 K, top face heated to T = 300 K.
        Checks that the 3-term adjoint sensitivities (elastic, RAMP expansion, conductivity)
        do not induce ill-conditioning in the linear solver (AMG) or non-positive definiteness.
        """
        nelx, nely, nelz = 8, 6, 6
        volfrac = 0.35
        delta_T = 300.0  # Delta T > 200 K

        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, penal=3.0, rmin=1.8,
            solver_type="amg", T_ref=0.0, alpha_th=1.0e-5
        )
        opt.fix_face("bottom")

        # Pure shear load on top face
        opt.add_load(nelx // 2, nely // 2, nelz, fy=200.0)

        # Transverse thermal gradient from bottom to top
        opt.fix_thermal_face("bottom", temp=0.0)
        opt.fix_thermal_face("top", temp=delta_T)

        res = opt.solve(max_iter=12, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")

        print("\n--- TEST 3: EXTREME SHEAR + TRANSVERSE THERMAL GRADIENT (Delta T = 300 K) ---")
        print(f"Iterations: {res.iterations_run}")
        print(f"Final Compliance: {res.compliance:.4e}")
        print(f"Final Volume Fraction: {res.volume_fraction:.4f}")

        self.assertTrue(res.success, "Shear + thermal gradient optimization must succeed.")
        self.assertTrue(np.all(np.isfinite(res.displacements)), "Displacement vector contains NaNs.")
        self.assertTrue(np.all(np.isfinite(res.temperatures)), "Temperature field contains NaNs.")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.02)
        self.assertIsNotNone(res.temperature_field, "temperature_field 3D array must be exposed.")
        self.assertEqual(res.temperature_field.shape, (nelz, nely, nelx))

    def test_stress4_opposing_thermal_contraction_delta_t_minus_250K(self):
        """
        Adversarial Test 4: Thermal Contraction (Delta T = -250 K) vs Outward Mechanical Tension.
        
        Tests negative temperature delta (cooling from T_ref = 300 K down to T = 50 K),
        generating inward thermal contraction forces while mechanical tip loads pull outward
        in +x tension (fx = +250.0).
        Verifies that RAMP negative load vector and 3-term adjoint sensitivities evaluate
        correctly without sign divergence or numerical stalling.
        """
        nelx, nely, nelz = 8, 6, 4
        volfrac = 0.30

        # T_ref = 300 K, boundaries at 50 K -> Delta T = -250 K
        opt = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            volfrac=volfrac, penal=3.0, rmin=1.8,
            solver_type="direct", T_ref=300.0, alpha_th=1.0e-5
        )
        opt.fix_face("left")
        # Outward mechanical tension
        opt.add_load(nelx, nely // 2, nelz // 2, fx=250.0)

        # Thermal boundary: cooling sink at right face
        opt.fix_thermal_face("left", temp=300.0)
        opt.fix_thermal_face("right", temp=50.0)  # -250 K delta

        res = opt.solve(max_iter=12, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")

        print("\n--- TEST 4: THERMAL CONTRACTION (Delta T = -250 K) VS TENSION ---")
        print(f"Iterations: {res.iterations_run}")
        print(f"Final Compliance: {res.compliance:.4e}")
        print(f"Min Temperature: {float(np.min(res.temperatures)):.2f} K")
        print(f"Final Volume Fraction: {res.volume_fraction:.4f}")

        self.assertTrue(res.success, "Thermal contraction solve must succeed.")
        self.assertTrue(np.all(np.isfinite(res.displacements)), "Displacements contain NaNs.")
        self.assertTrue(np.all(np.isfinite(res.temperatures)), "Temperatures contain NaNs.")
        self.assertAlmostEqual(res.volume_fraction, volfrac, delta=0.02)
        self.assertLessEqual(float(np.min(res.temperatures)), 60.0, "Cold boundary must reach ~50 K.")

    def test_stress5_anisotropic_mesh_length_scale_suppression(self):
        """
        Adversarial Test 5: Length-Scale Suppression on Anisotropic Mesh (dx != dy != dz).
        
        Evaluates minimum length-scale feature control when grid spacing is non-uniform:
          dx = 1.0, dy = 1.0, dz = 1.5  ==> max(dx, dy, dz) = 1.5.
          Domain: 10 x 8 x 8 (Lx = 10.0, Ly = 8.0, Lz = 12.0)
        
        Comparing:
          Unconstrained: rmin = 0.9 * max_dim = 1.35
          Constrained:   rmin = 2.0 * max_dim = 3.0
        
        Asserts that 1-voxel thin branches are demonstrably suppressed by the constrained filter.
        """
        nelx, nely, nelz = 10, 8, 8
        dx, dy, dz = 1.0, 1.0, 1.5
        max_dim = max(dx, dy, dz)
        volfrac = 0.35

        # Run A: Unconstrained
        rmin_un = 0.9 * max_dim
        opt_un = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            dx=dx, dy=dy, dz=dz,
            volfrac=volfrac, penal=3.0, rmin=rmin_un,
            solver_type="direct", T_ref=0.0
        )
        opt_un.fix_face("left")
        opt_un.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        opt_un.fix_thermal_face("left", temp=0.0)
        opt_un.fix_thermal_face("right", temp=100.0)
        res_un = opt_un.solve(max_iter=15, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")

        # Run B: Constrained
        rmin_con = 2.0 * max_dim
        opt_con = SIMPOptimizer3D(
            nelx=nelx, nely=nely, nelz=nelz,
            dx=dx, dy=dy, dz=dz,
            volfrac=volfrac, penal=3.0, rmin=rmin_con,
            solver_type="direct", T_ref=0.0
        )
        opt_con.fix_face("left")
        opt_con.add_load(nelx, nely // 2, nelz // 2, fz=-100.0)
        opt_con.fix_thermal_face("left", temp=0.0)
        opt_con.fix_thermal_face("right", temp=100.0)
        res_con = opt_con.solve(max_iter=15, tol=1e-3, mode="thermo_elastic", optimizer_type="mma")

        metrics_un = compute_3d_thin_fraction(res_un.density_matrix, threshold=0.35)
        metrics_con = compute_3d_thin_fraction(res_con.density_matrix, threshold=0.35)

        print("\n--- TEST 5: ANISOTROPIC MESH LENGTH SCALE COMPARISON ---")
        print(f"Unconstrained (rmin={rmin_un:.2f}): Solid={metrics_un['solid_count']}, Eroded={metrics_un['eroded_count']}, "
              f"Thin={metrics_un['thin_fraction']:.2%}, Core={metrics_un['retained_core_fraction']:.2%}")
        print(f"Constrained   (rmin={rmin_con:.2f}): Solid={metrics_con['solid_count']}, Eroded={metrics_con['eroded_count']}, "
              f"Thin={metrics_con['thin_fraction']:.2%}, Core={metrics_con['retained_core_fraction']:.2%}")

        self.assertLess(
            metrics_con["thin_fraction"], metrics_un["thin_fraction"],
            f"Anisotropic constrained thin fraction ({metrics_con['thin_fraction']:.2%}) was not less than unconstrained ({metrics_un['thin_fraction']:.2%})."
        )
        self.assertGreater(
            metrics_con["retained_core_fraction"], metrics_un["retained_core_fraction"],
            f"Anisotropic constrained core ({metrics_con['retained_core_fraction']:.2%}) was not greater than unconstrained ({metrics_un['retained_core_fraction']:.2%})."
        )


if __name__ == "__main__":
    unittest.main()
