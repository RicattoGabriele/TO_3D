import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional

# Ensure project root and dashboard are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in [str(PROJECT_ROOT), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics_engine.simp_engine_3d import SIMPOptimizer3D, SIMPResult3D
from physics_engine.simp_engine_2d import SIMPOptimizer2D

# Optional PicoGK integration
try:
    from dashboard.core.execution_runner import run_generation
    from dashboard.core.visualizer import compute_mesh_metrics, render_3d_mesh_plotly
    PICOGK_AVAILABLE = True
except Exception:
    PICOGK_AVAILABLE = False

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CE-3D: Multi-Objective Topology Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400;1,700&display=swap');
    * {
        font-family: 'Courier Prime', monospace !important;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">CE-3D: Multi-Objective Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Continuum Topology Optimization for Structural Compliance and Linearized Buckling Stability</div>', unsafe_allow_html=True)

tab_3d, tab_2d, tab_docs, tab_roadmap = st.tabs([
    "🧊 3D Continuum Engine",
    "📐 2D Engine & Pareto Sweep",
    "📖 Theoretical Log & PDF",
    "🚀 Multi-Physics Roadmap (Noyron)"
])

# =============================================================================
# TAB 1: 3D Continuum Optimization
# =============================================================================
with tab_3d:
    col_cfg, col_vis = st.columns([1, 2])

    with col_cfg:
        st.subheader("1. 3D Domain Discretization")
        c1, c2, c3 = st.columns(3)
        nelx = c1.number_input("Elements X", min_value=6, max_value=80, value=20, step=2)
        nely = c2.number_input("Elements Y", min_value=4, max_value=50, value=10, step=2)
        nelz = c3.number_input("Elements Z", min_value=4, max_value=50, value=10, step=2)

        c4, c5, c6 = st.columns(3)
        dx = c4.number_input("dx (mm)", min_value=0.2, max_value=10.0, value=1.0, step=0.5)
        dy = c5.number_input("dy (mm)", min_value=0.2, max_value=10.0, value=1.0, step=0.5)
        dz = c6.number_input("dz (mm)", min_value=0.2, max_value=10.0, value=1.0, step=0.5)

        total_voxels = nelx * nely * nelz
        st.caption(f"Grid: {total_voxels:,} voxels ({total_voxels * 12 / 1024:.1f} MB RAM estimate)")

        st.subheader("2. Optimization Objectives")
        volfrac = st.slider("Target Volume Fraction (Vf)", 0.10, 0.80, 0.35, 0.05)
        alpha = st.slider(
            "Buckling Weight Factor (α)", 0.00, 0.90, 0.30, 0.05,
            help="0.0 = Minimum Compliance (maximum stiffness). 0.9 = High Buckling Stability (prevents slender member collapse)."
        )
        rmin = st.slider("Filter Radius r_min (mm)", 1.0, 5.0, 1.5, 0.5)
        max_iter = st.slider("Max Iterations", 5, 50, 25, 5)
        solver_type = st.selectbox("Linear Equation Solver", ["pcg (Jacobi PCG, Low Memory)", "direct (SuperLU)"])
        solver_key = "pcg" if "pcg" in solver_type else "direct"

        st.subheader("3. Boundary Conditions & Loads")
        bc_preset = st.selectbox("Support Preset", ["Cantilever (Clamped Left Face)", "Bridge (Bottom Corners)", "Center Beam"])
        load_mag = st.number_input("Applied Load Magnitude (N)", value=100.0, step=10.0)

        run_3d_btn = st.button("⚡ Solve 3D Topology Optimization", type="primary", use_container_width=True)

    with col_vis:
        progress_box = st.empty()
        status_box = st.empty()
        vis_box = st.empty()
        chart_box = st.empty()

    if run_3d_btn:
        progress_bar = progress_box.progress(0)

        opt3d = SIMPOptimizer3D(
            nelx=int(nelx), nely=int(nely), nelz=int(nelz),
            dx=float(dx), dy=float(dy), dz=float(dz),
            E0=1.0, Emin=1e-9, nu=0.3,
            penal=3.0, penal_g=6.0,
            rmin=float(rmin), volfrac=float(volfrac),
            solver_type=solver_key
        )

        if bc_preset == "Cantilever (Clamped Left Face)":
            opt3d.fix_face("left", fix_x=True, fix_y=True, fix_z=True)
            opt3d.add_load(nelx, 0, nelz // 2, fx=0.0, fy=-float(load_mag), fz=0.0)
        elif bc_preset == "Bridge (Bottom Corners)":
            opt3d.fix_face("bottom", fix_x=True, fix_y=True, fix_z=True)
            opt3d.add_load(nelx // 2, nely, nelz // 2, fx=0.0, fy=-float(load_mag), fz=0.0)
        else:
            opt3d.fix_face("left", fix_x=True, fix_y=True, fix_z=True)
            opt3d.add_load(nelx, nely // 2, nelz // 2, fx=0.0, fy=-float(load_mag), fz=0.0)

        def on_3d_progress(it, max_it, comp=0.0, vol=0.0, *args):
            pct = int((it / max_it) * 100)
            progress_bar.progress(pct)
            status_box.info(f"Iter {it:02d}/{max_it:02d} | Compliance: {comp:.3e} | Vol: {vol*100:.1f}%")

        mode = "buckling_max" if alpha > 0.0 else "compliance"
        start_t = time.time()
        res3d = opt3d.solve(
            max_iter=int(max_iter),
            tol=0.015,
            mode=mode,
            alpha_buckling=float(alpha),
            progress_callback=on_3d_progress
        )
        elapsed = time.time() - start_t

        st.session_state["res3d"] = res3d
        st.session_state["opt3d"] = opt3d
        status_box.success(f"✅ Converged in {res3d.iterations_run} iterations ({elapsed:.2f} s) | Peak RAM: {res3d.peak_memory_mb:.1f} MB")

    if "res3d" in st.session_state:
        res = st.session_state["res3d"]
        opt = st.session_state["opt3d"]

        # Convergence plot
        if res.compliance_history:
            fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
            iters = list(range(1, len(res.compliance_history) + 1))
            fig_conv.add_trace(
                go.Scatter(x=iters, y=res.compliance_history, name="Compliance (Strain Energy)", line=dict(color="#3b82f6", width=2)),
                secondary_y=False
            )
            if res.blf_history:
                fig_conv.add_trace(
                    go.Scatter(x=iters, y=res.blf_history, name="Buckling Load Factor (BLF)", line=dict(color="#ef4444", width=2, dash="dash")),
                    secondary_y=True
                )
            fig_conv.update_layout(
                title_text="Convergence History",
                height=240,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", y=1.15)
            )
            chart_box.plotly_chart(fig_conv, use_container_width=True)

        # 3D Isosurface
        thresh = st.slider("Relative Density Threshold", 0.1, 0.9, 0.35, 0.05, key="iso_th")
        X, Y, Z = np.mgrid[0:opt.nelx*opt.dx:complex(0, opt.nelx),
                           0:opt.nely*opt.dy:complex(0, opt.nely),
                           0:opt.nelz*opt.dz:complex(0, opt.nelz)]

        fig3d = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=res.density_matrix.flatten(),
            isomin=thresh, isomax=1.0,
            surface_count=3,
            colorscale='Blues_r',
            caps=dict(x_show=True, y_show=True, z_show=True),
            colorbar=dict(title="Density ρ", len=0.6)
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            height=480
        )
        vis_box.plotly_chart(fig3d, use_container_width=True)

        # Export STL
        st.subheader("Watertight STL Export")
        c_stl1, c_stl2 = st.columns([2, 1])
        with c_stl1:
            st.write("Generates a manifold, closed boundary surface STL ready for 3D printing or CAD smoothing.")
        with c_stl2:
            if st.button("💾 Generate & Download STL", use_container_width=True):
                tmp_stl = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                tmp_stl.close()
                opt.export_stl(res, filepath=tmp_stl.name, threshold=thresh)
                with open(tmp_stl.name, "rb") as f:
                    stl_data = f.read()
                st.download_button(
                    "Click to Download STL", data=stl_data,
                    file_name="CE_3D_Optimized_Structure.stl", mime="application/sla"
                )

# =============================================================================
# TAB 2: 2D Continuum Engine & Pareto Sweep
# =============================================================================
with tab_2d:
    col2d_1, col2d_2 = st.columns([1, 2])
    with col2d_1:
        st.subheader("2D Fast Exploration")
        nelx2 = st.number_input("Elements X (2D)", 10, 200, 40, 2)
        nely2 = st.number_input("Elements Y (2D)", 6, 100, 20, 2)
        volfrac2 = st.slider("Target Volume Fraction (2D)", 0.1, 0.8, 0.4, 0.05)
        run_sweep = st.button("📊 Run 11-Step Pareto Alpha Sweep (α = 0.0 → 1.0)")

    with col2d_2:
        if run_sweep:
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            st.markdown("#### Pareto Frontier Exploration (Compliance vs Buckling)")
            sweep_prog = st.progress(0)

            sweep_cols = st.columns(3)
            placeholders = [sweep_cols[i % 3].empty() for i in range(len(alphas))]

            for i, a in enumerate(alphas):
                opt2d = SIMPOptimizer2D(
                    nelx=int(nelx2), nely=int(nely2), dx=2.0, dy=2.0,
                    penal=3.0, rmin=2.5, volfrac=float(volfrac2)
                )
                opt2d.fix_wall("left")
                opt2d.add_load(int(nelx2), 0, fx=0.0, fy=-500.0)

                res_sw = opt2d.solve(max_iter=30, tol=0.015, mode="buckling_max", alpha_buckling=a)
                with placeholders[i].container():
                    st.markdown(f"**α = {a:.1f}**")
                    blf_str = f"{res_sw.blf_history[-1]:.3f}" if res_sw.blf_history else "N/A"
                    st.caption(f"C: {res_sw.compliance:.1f} | BLF: {blf_str}")
                    fig_2d = opt2d.to_scientific_figures(res=res_sw, threshold=0.40, view_style="silhouette_bw", smooth=True)
                    fig_2d.update_layout(height=200, margin=dict(l=5, r=5, t=5, b=5))
                    st.plotly_chart(fig_2d, use_container_width=True)
                sweep_prog.progress((i + 1) / len(alphas))

# =============================================================================
# TAB 3: Theoretical Documentation & Citations
# =============================================================================
with tab_docs:
    st.subheader("Academic Formulation & Log of Changes")
    st.markdown("""
    Every equation and theoretical adaptation from 2D to 3D is documented in **`docs/TO_3D_log.pdf`**, mirroring the style of the original paper:

    1. **Element Formulation**: 8-node hexahedral isoparametric elements ($H8$) with trilinear shape functions:
       $$N_i(\\xi, \\eta, \\zeta) = \\frac{1}{8}(1 + \\xi_i \\xi)(1 + \\eta_i \\eta)(1 + \\zeta_i \\zeta)$$
    2. **Linearized Buckling Stability**: Generalized eigenvalue problem:
       $$(\\mathbf{K} + \\lambda_i \\mathbf{G})\\boldsymbol{\\phi}_i = \\mathbf{0}, \\quad \\mu_i = \\frac{1}{\\lambda_i}$$
    3. **6-Basis Geometric Stiffness Decomposition**:
       $$\\mathbf{G}_e = \\sum_{k=1}^6 \\sigma_{e, k} \\mathbf{G}_{0, k}$$
       eliminating numerical quadrature inside the optimization iterations.
    4. **Adjoint Sensitivities & Dynamic Scaling**:
       $$\\frac{\\partial F^{(i)}}{\\partial x_e} = (1 - \\alpha) \\frac{1}{S_C^{(i)}} \\frac{\\partial C^{(i)}}{\\partial x_e} + \\alpha \\frac{1}{S_\\mu^{(i)}} \\frac{\\partial \\mu_1^{(i)}}{\\partial x_e}$$
    """)

    pdf_path = PROJECT_ROOT / "docs" / "TO_3D_log.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Theoretical Log (PDF, 540 KB)",
                data=f.read(),
                file_name="TO_3D_log.pdf",
                mime="application/pdf"
            )

# =============================================================================
# TAB 4: Multi-Physics Roadmap (Toward Leap71 Noyron)
# =============================================================================
with tab_roadmap:
    st.subheader("The Leap71 Noyron Paradigm: Multi-Field Generative Physics")
    st.markdown("""
    The goal of this platform is to scale beyond structural optimization into **coupled multi-physics computational engineering**:

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Multi-Physics Objective Coupling                │
    │  min F = w_mech * Compliance + w_stab * (1/BLF) + w_th * Thermal│
    └──────────────┬───────────────────────────────┬──────────────────┘
                   │                               │
           ┌───────▼───────┐               ┌───────▼───────┐
           │ 3D Elasticity │               │ 3D Heat Flow  │
           │  K(x) U = F   │               │ K_th(x) T = Q │
           └───────┬───────┘               └───────┬───────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                   ┌───────────────▼───────────────┐
                   │  Combined Adjoint Sensitivity │
                   └───────────────┬───────────────┘
                                   │
                   ┌───────────────▼───────────────┐
                   │  PicoGK Implicit Smoothing &  │
                   │  Watertight 3D Solid Output   │
                   └───────────────────────────────┘
    ```

    ### Planned Next Step: 3D Steady-State Heat Conduction
    - Poisson equation $\\nabla \\cdot (k(\\mathbf{x}) \\nabla T) + q = 0$ on the identical H8 voxel grid.
    - Thermal compliance minimization $\\mathbf{Q}^T \\mathbf{T}$ to design optimal heat-dissipating cooling channels and fin structures.
    - Fusing structural load paths with thermal conductive paths into a single optimal geometry.
    """)
