"""
Example: 3D Topology Optimization of a Cantilever Beam.
Demonstrates:
- Setting up a 3D design domain (20x10x10 voxels)
- Applying fixed face boundary conditions
- Applying a transverse point load
- Running multi-objective optimization (compliance + buckling resistance)
- Exporting the resulting optimized geometry directly to an STL file
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physics_engine.simp_engine_3d import SIMPOptimizer3D

def main():
    print("=" * 70)
    print("3D CONTINUUM TOPOLOGY OPTIMIZATION DEMO")
    print("Multi-Objective: Compliance Minimization + Linearized Buckling Stability")
    print("=" * 70)

    # 1. Initialize optimizer
    nelx, nely, nelz = 20, 10, 10
    dx, dy, dz = 1.0, 1.0, 1.0
    volfrac = 0.35
    rmin = 1.5

    print(f"\n[1/4] Initializing Domain: {nelx}x{nely}x{nelz} elements ({nelx*nely*nelz:,} voxels)")
    print(f"      Target Volume Fraction: {volfrac*100:.0f}%, Filter Radius: {rmin} mm")
    
    opt = SIMPOptimizer3D(
        nelx=nelx, nely=nely, nelz=nelz,
        dx=dx, dy=dy, dz=dz,
        E0=1.0, Emin=1e-9, nu=0.3,
        penal=3.0, penal_g=6.0,
        rmin=rmin, volfrac=volfrac,
        solver_type="pcg"
    )

    # 2. Boundary conditions
    print("\n[2/4] Applying Boundary Conditions...")
    # Fix the entire left face (x = 0)
    opt.fix_face(face="left", fix_x=True, fix_y=True, fix_z=True)
    
    # Apply downward tip load at the bottom edge of the right face (x = nelx, y = 0, z = nelz/2)
    opt.add_load(i=nelx, j=0, k=nelz // 2, fx=0.0, fy=-100.0, fz=0.0)
    print(f"      Fixed face 'left' (3 DOFs per node)")
    print(f"      Point load F=(0, -100, 0) N at (x={nelx}, y=0, z={nelz//2})")

    # 3. Solve optimization
    print("\n[3/4] Solving Topology Optimization Loop...")
    start_t = time.time()
    
    def on_progress(it, max_it, comp=0.0, vol=0.0, *args):
        print(f"      Iter {it:02d}/{max_it:02d} | Compliance = {comp:.4e} | Mean Density = {vol*100:.1f}%")

    res = opt.solve(
        max_iter=25,
        tol=0.015,
        mode="buckling_max",
        alpha_buckling=0.30,
        progress_callback=on_progress
    )
    elapsed = time.time() - start_t

    print("\n" + "=" * 70)
    print(f"OPTIMIZATION COMPLETE ({elapsed:.2f} s)")
    print(f"Status:             {res.status_message}")
    print(f"Iterations:         {res.iterations_run}")
    print(f"Final Compliance:   {res.compliance:.4e}")
    if res.blf_history:
        print(f"Fundamental BLF:    {res.blf_history[-1]:.4f}")
    print(f"Final Volume Frac:  {res.volume_fraction*100:.2f}%")
    print("=" * 70)

    # 4. Export to STL
    output_stl = PROJECT_ROOT / "outputs" / "cantilever_3d_optimized.stl"
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[4/4] Exporting Watertight STL to '{output_stl.name}'...")
    stl_path = opt.export_stl(res, filepath=str(output_stl), threshold=0.35)
    print(f"      Saved STL successfully at: {stl_path}")
    print(f"      File size: {Path(stl_path).stat().st_size:,} bytes")
    print("\nDemo completed successfully! You can inspect the STL in MeshLab, Blender, or 3D viewer.")

if __name__ == "__main__":
    main()
