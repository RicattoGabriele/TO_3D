import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

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
import trimesh

from physics_engine.simp_engine_3d import SIMPOptimizer3D, SIMPResult3D

# -----------------------------------------------------------------------------
# Configuration & Minimal Monochrome Dark Styling (Inconsolata Light 300)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CE-3D",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@300;400;500;600&display=swap');

    /* Global Inconsolata 300 & Pure Black Canvas */
    *, html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, label, input, button, select {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 300 !important;
    }

    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Completely hide Streamlit Header, Toolbar & Deploy Button */
    header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #18181b !important;
    }
    section[data-testid="stSidebar"] * {
        font-family: 'Inconsolata', monospace !important;
    }

    /* Section Headers */
    .sec-head {
        font-size: 0.80rem;
        font-weight: 500 !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a1a1aa;
        border-bottom: 1px solid #18181b;
        padding-bottom: 3px;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }

    /* Radio Buttons - Zero Orange, Pure White Checked */
    div[data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    div[data-testid="stRadio"] label {
        color: #ffffff !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stRadio"] div[role="radio"] {
        background-color: transparent !important;
        border: 1px solid #3f3f46 !important;
    }
    div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] {
        border-color: #ffffff !important;
        background-color: transparent !important;
    }
    div[data-testid="stRadio"] div[role="radio"] div {
        background-color: transparent !important;
    }
    div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] div {
        background-color: #ffffff !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:checked {
        accent-color: #ffffff !important;
    }
    div[data-testid="stRadio"] svg {
        fill: #ffffff !important;
    }

    /* Checkboxes */
    div[data-testid="stCheckbox"] label {
        color: #ffffff !important;
    }
    div[data-testid="stCheckbox"] div[role="checkbox"] {
        background-color: transparent !important;
        border: 1px solid #3f3f46 !important;
    }
    div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
    }
    div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] svg {
        fill: #000000 !important;
    }
    div[data-testid="stCheckbox"] input[type="checkbox"] {
        accent-color: #ffffff !important;
    }

    /* Sliders - Zero Orange, Pure White Active */
    div[data-testid="stSlider"] * {
        accent-color: #ffffff !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 1px solid #000000 !important;
        box-shadow: none !important;
        width: 14px !important;
        height: 14px !important;
    }
    div[data-baseweb="slider"] > div > div:first-child {
        background: #18181b !important;
    }
    div[data-baseweb="slider"] div[style*="background-color"] {
        background-color: #ffffff !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #52525b !important;
        font-size: 0.72rem !important;
    }

    /* Inputs & Selectboxes */
    input, textarea, [data-baseweb="input"], [data-baseweb="base-input"] {
        background-color: #09090b !important;
        color: #ffffff !important;
        border: 1px solid #27272a !important;
        border-radius: 0 !important;
    }
    input:focus, textarea:focus {
        border-color: #ffffff !important;
        outline: none !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #09090b !important;
        border: 1px solid #27272a !important;
        border-radius: 0 !important;
    }
    div[data-baseweb="select"] * {
        color: #ffffff !important;
        background-color: #09090b !important;
    }
    li[role="option"] {
        background-color: #09090b !important;
        color: #ffffff !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #27272a !important;
        color: #ffffff !important;
    }
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: #09090b !important;
        color: #ffffff !important;
        border: 1px solid #27272a !important;
    }

    /* Buttons */
    .stButton > button {
        letter-spacing: 0.1em;
        text-transform: uppercase;
        background-color: #050505 !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
        border-radius: 0 !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.12s ease !important;
    }
    .stButton > button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
    }
    button[kind="primary"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ffffff !important;
        font-weight: 500 !important;
    }
    button[kind="primary"]:hover {
        background-color: #d4d4d8 !important;
        color: #000000 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-weight: 500 !important;
        color: #ffffff !important;
        font-size: 1.10rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 300 !important;
        color: #71717a !important;
        font-size: 0.70rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #ffffff !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #050505 !important;
        border: 1px solid #18181b !important;
        color: #71717a !important;
    }

    /* Kill focus rings & outlines */
    *:focus {
        outline: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3D Domain Preview Function (Delicate Micro-Markers, Wireframe Box, Exact CAD Mesh)
# -----------------------------------------------------------------------------
def build_domain_preview_figure(
    nelx: int, nely: int, nelz: int,
    dx: float, dy: float, dz: float,
    fixed_nodes_coords: List[Tuple[float, float, float, str]],
    applied_loads: List[Tuple[float, float, float, float, float, float]],
    res: Optional[SIMPResult3D] = None,
    threshold: float = 0.35,
    view_mode: str = "preview",
    smooth_iterations: int = 0,
    stress_field_key: str = "von_mises",
    stress_colormap: str = "Turbo",
    stress_field_name: str = "Von Mises"
) -> go.Figure:
    Lx = float(nelx * dx)
    Ly = float(nely * dy)
    Lz = float(nelz * dz)

    fig = go.Figure()

    # 1. Bounding Box Wireframe (Clean 1.5px lines)
    fig.add_trace(go.Scatter3d(
        x=[
            0, Lx, Lx, 0, 0, None,
            0, Lx, Lx, 0, 0, None,
            0, 0, None,
            Lx, Lx, None,
            Lx, Lx, None,
            0, 0
        ],
        y=[
            0, 0, Ly, Ly, 0, None,
            0, 0, Ly, Ly, 0, None,
            0, 0, None,
            0, 0, None,
            Ly, Ly, None,
            Ly, Ly
        ],
        z=[
            0, 0, 0, 0, 0, None,
            Lz, Lz, Lz, Lz, Lz, None,
            0, Lz, None,
            0, Lz, None,
            0, Lz, None,
            0, Lz
        ],
        mode="lines",
        line=dict(color="#52525b", width=1.5),
        name="Domain",
        hoverinfo="skip",
        showlegend=False
    ))

    # Faint domain shading
    fig.add_trace(go.Mesh3d(
        x=[0, Lx, Lx, 0, 0, Lx, Lx, 0],
        y=[0, 0, Ly, Ly, 0, 0, Ly, Ly],
        z=[0, 0, 0, 0, Lz, Lz, Lz, Lz],
        i=[0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 2, 2],
        j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 3, 7],
        k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 7, 6],
        color="#27272a",
        opacity=0.03,
        name="Volume",
        hoverinfo="skip",
        showlegend=False
    ))

    # 2. Fixed Support Markers (Small, discrete cyan squares: size=3.5)
    if fixed_nodes_coords:
        fx_x = [pt[0] for pt in fixed_nodes_coords]
        fx_y = [pt[1] for pt in fixed_nodes_coords]
        fx_z = [pt[2] for pt in fixed_nodes_coords]
        fx_hover = [f"Support ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}) mm<br>DOFs: {pt[3]}" for pt in fixed_nodes_coords]

        fig.add_trace(go.Scatter3d(
            x=fx_x, y=fx_y, z=fx_z,
            mode="markers",
            marker=dict(
                size=3.5,
                color="#00f5d4",
                symbol="square",
                opacity=0.95
            ),
            text=fx_hover,
            hoverinfo="text",
            name="Supports",
            showlegend=False
        ))

    # 3. Applied Loads (Small magenta diamond: size=4, 2px arrow)
    if applied_loads:
        for idx, (lx, ly, lz, f_x, f_y, f_z) in enumerate(applied_loads):
            mag = np.sqrt(f_x**2 + f_y**2 + f_z**2)
            if mag < 1e-6:
                continue

            fig.add_trace(go.Scatter3d(
                x=[lx], y=[ly], z=[lz],
                mode="markers",
                marker=dict(
                    size=4.0,
                    color="#ff0055",
                    symbol="diamond",
                    opacity=1.0
                ),
                text=[f"Load #{idx+1}: ({f_x:.1f}, {f_y:.1f}, {f_z:.1f}) N<br>Point: ({lx:.1f}, {ly:.1f}, {lz:.1f}) mm"],
                hoverinfo="text",
                name=f"Load #{idx+1}",
                showlegend=False
            ))

            arrow_scale = 0.15 * max(Lx, Ly, Lz)
            dir_x = (f_x / mag) * arrow_scale
            dir_y = (f_y / mag) * arrow_scale
            dir_z = (f_z / mag) * arrow_scale

            fig.add_trace(go.Scatter3d(
                x=[lx, lx + dir_x],
                y=[ly, ly + dir_y],
                z=[lz, lz + dir_z],
                mode="lines+markers",
                line=dict(color="#ff0055", width=2.5),
                marker=dict(size=[0, 4], color="#ff0055", symbol="diamond"),
                text=[None, f"F = {mag:.1f} N"],
                hoverinfo="text",
                name=f"Load Vector #{idx+1}",
                showlegend=False
            ))

    # 4. Topology Mesh (Exact Watertight CAD Faces via Trimesh Extraction + Taubin Smoothing)
    if res is not None and view_mode in ["tension_compression", "result", "cad"]:
        try:
            m = res.get_boundary_mesh(threshold=threshold, smooth_iterations=smooth_iterations)
            if len(m.vertices) > 0 and len(m.faces) > 0:
                vx, vy, vz = m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2]

                if view_mode == "tension_compression" and res.stresses is not None and "sigma_signed" in res.stresses:
                    # Map signed stress to vertices (positive = tension/red, negative = compression/blue)
                    s_signed = res.stresses["sigma_signed"]
                    try:
                        from scipy.ndimage import map_coordinates
                        coords = np.array([vz / dz - 0.5, vy / dy - 0.5, vx / dx - 0.5])
                        v_stress = map_coordinates(s_signed, coords, order=1, mode='nearest')
                    except Exception:
                        ix = np.clip(np.floor(vx / dx).astype(int), 0, nelx - 1)
                        iy = np.clip(np.floor(vy / dy).astype(int), 0, nely - 1)
                        iz = np.clip(np.floor(vz / dz).astype(int), 0, nelz - 1)
                        v_stress = s_signed[iz, iy, ix]
                    max_abs = float(np.max(np.abs(v_stress)))
                    if max_abs < 1e-6:
                        max_abs = 1.0

                    fig.add_trace(go.Mesh3d(
                        x=vx, y=vy, z=vz,
                        i=m.faces[:, 0], j=m.faces[:, 1], k=m.faces[:, 2],
                        intensity=v_stress,
                        colorscale="RdBu_r",  # Blue = Compression, Red = Tension
                        cmin=-max_abs,
                        cmax=max_abs,
                        colorbar=dict(
                            title=dict(text="σ (MPa)<br>[- Comp / + Tens]", font=dict(color="#ffffff", size=9, family="Inconsolata")),
                            tickfont=dict(color="#71717a", size=8, family="Inconsolata"),
                            len=0.50,
                            x=1.02
                        ),
                        opacity=0.98,
                        flatshading=(smooth_iterations == 0),
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                        name="Tension / Compression",
                        showlegend=False
                    ))
                elif view_mode == "cad":
                    fig.add_trace(go.Mesh3d(
                        x=vx, y=vy, z=vz,
                        i=m.faces[:, 0], j=m.faces[:, 1], k=m.faces[:, 2],
                        color="#e4e4e7",  # Sleek titanium white
                        opacity=0.96,
                        flatshading=(smooth_iterations == 0),
                        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25),
                        name="CAD Surface",
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Mesh3d(
                        x=vx, y=vy, z=vz,
                        i=m.faces[:, 0], j=m.faces[:, 1], k=m.faces[:, 2],
                        color="#38bdf8",
                        opacity=0.96,
                        flatshading=(smooth_iterations == 0),
                        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.2),
                        name="Topology",
                        showlegend=False
                    ))
        except Exception:
            pass

    # 5. Thermal Stress Heatmap (High-Contrast Gradient on Solid Voxels)
    if res is not None and view_mode == "stress" and res.stresses is not None:
        stress_matrix = res.stresses.get(stress_field_key, res.stresses.get("von_mises"))
        solid_mask = (res.density_matrix >= threshold)
        iz_arr, iy_arr, ix_arr = np.where(solid_mask)

        if len(ix_arr) > 0:
            solid_x = (ix_arr + 0.5) * dx
            solid_y = (iy_arr + 0.5) * dy
            solid_z = (iz_arr + 0.5) * dz
            solid_vals = stress_matrix[iz_arr, iy_arr, ix_arr]
            solid_dens = res.density_matrix[iz_arr, iy_arr, ix_arr]

            hover_texts = [
                f"Voxel ({ix},{iy},{iz}) | ρ={rho:.2f}<br>{stress_field_name}: {val:.2f} MPa"
                for ix, iy, iz, rho, val in zip(ix_arr, iy_arr, iz_arr, solid_dens, solid_vals)
            ]

            marker_sz = max(3, min(8, int(180 / max(nelx, nely, nelz))))

            fig.add_trace(go.Scatter3d(
                x=solid_x, y=solid_y, z=solid_z,
                mode="markers",
                marker=dict(
                    size=marker_sz,
                    color=solid_vals,
                    colorscale=stress_colormap,
                    colorbar=dict(
                        title=dict(text=f"{stress_field_name} (MPa)", font=dict(color="#ffffff", size=9, family="Inconsolata")),
                        tickfont=dict(color="#71717a", size=8, family="Inconsolata"),
                        len=0.50,
                        x=1.02
                    ),
                    showscale=True,
                    symbol="square",
                    opacity=0.95
                ),
                text=hover_texts,
                hoverinfo="text",
                name=stress_field_name,
                showlegend=False
            ))

    # Plotly Layout: Pitch Black Background, Minimalist Grid
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        scene=dict(
            xaxis=dict(
                title=dict(text="X (mm)", font=dict(color="#71717a", size=10, family="Inconsolata")),
                tickfont=dict(color="#52525b", size=8, family="Inconsolata"),
                backgroundcolor="#000000",
                gridcolor="#18181b",
                zerolinecolor="#27272a",
                range=[-0.05 * Lx, 1.15 * Lx]
            ),
            yaxis=dict(
                title=dict(text="Y (mm)", font=dict(color="#71717a", size=10, family="Inconsolata")),
                tickfont=dict(color="#52525b", size=8, family="Inconsolata"),
                backgroundcolor="#000000",
                gridcolor="#18181b",
                zerolinecolor="#27272a",
                range=[-0.05 * Ly, 1.15 * Ly]
            ),
            zaxis=dict(
                title=dict(text="Z (mm)", font=dict(color="#71717a", size=10, family="Inconsolata")),
                tickfont=dict(color="#52525b", size=8, family="Inconsolata"),
                backgroundcolor="#000000",
                gridcolor="#18181b",
                zerolinecolor="#27272a",
                range=[-0.05 * Lz, 1.15 * Lz]
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.8, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )

    return fig


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="font-size:1.1rem; font-weight:500; letter-spacing:0.18em; color:#ffffff; margin-bottom:0.8rem;">CE-3D</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">1. Mesh & Dimensioni</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    nelx = c1.number_input("nelx", min_value=4, max_value=80, value=20, step=2)
    nely = c2.number_input("nely", min_value=2, max_value=50, value=10, step=2)
    nelz = c3.number_input("nelz", min_value=2, max_value=50, value=10, step=2)

    cs1, cs2, cs3 = st.columns(3)
    dx = cs1.number_input("dx", min_value=0.1, max_value=20.0, value=1.0, step=0.5)
    dy = cs2.number_input("dy", min_value=0.1, max_value=20.0, value=1.0, step=0.5)
    dz = cs3.number_input("dz", min_value=0.1, max_value=20.0, value=1.0, step=0.5)

    Lx = nelx * dx
    Ly = nely * dy
    Lz = nelz * dz

    st.markdown('<div class="sec-head">2. Vincoli</div>', unsafe_allow_html=True)
    support_mode = st.radio(
        "Vincoli:",
        ["Incastro x=0", "Appoggio base", "Punto singolo"],
        index=0,
        label_visibility="collapsed"
    )

    sup_x, sup_y, sup_z = 0.0, 0.0, 0.0
    fix_u, fix_v, fix_w = True, True, True
    custom_supports: List[Tuple[float, float, float, str]] = []

    if support_mode == "Incastro x=0":
        for j_idx in range(0, nely + 1, max(1, nely // 3)):
            for k_idx in range(0, nelz + 1, max(1, nelz // 3)):
                custom_supports.append((0.0, j_idx * dy, k_idx * dz, "Ux, Uy, Uz"))
    elif support_mode == "Appoggio base":
        custom_supports.append((0.0, 0.0, 0.0, "Ux, Uy, Uz"))
        custom_supports.append((0.0, 0.0, Lz, "Ux, Uy, Uz"))
        custom_supports.append((Lx, 0.0, 0.0, "Uy, Uz"))
        custom_supports.append((Lx, 0.0, Lz, "Uy, Uz"))
    else:
        col_sx, col_sy, col_sz = st.columns(3)
        sup_x = col_sx.number_input("X (mm)", 0.0, float(Lx), 0.0, float(dx))
        sup_y = col_sy.number_input("Y (mm)", 0.0, float(Ly), 0.0, float(dy))
        sup_z = col_sz.number_input("Z (mm)", 0.0, float(Lz), 0.0, float(dz))

        cd1, cd2, cd3 = st.columns(3)
        fix_u = cd1.checkbox("Ux", value=True)
        fix_v = cd2.checkbox("Uy", value=True)
        fix_w = cd3.checkbox("Uz", value=True)
        dof_desc = []
        if fix_u: dof_desc.append("Ux")
        if fix_v: dof_desc.append("Uy")
        if fix_w: dof_desc.append("Uz")
        custom_supports.append((sup_x, sup_y, sup_z, ", ".join(dof_desc) if dof_desc else "None"))

    st.markdown('<div class="sec-head">3. Carichi</div>', unsafe_allow_html=True)
    load_mode = st.radio(
        "Carichi:",
        ["Carico estremità", "Carico puntuale"],
        index=0,
        label_visibility="collapsed"
    )

    applied_loads: List[Tuple[float, float, float, float, float, float]] = []
    if load_mode == "Carico estremità":
        fy_mag = st.number_input("Fy (N)", value=-100.0, step=10.0)
        applied_loads.append((Lx, 0.0, Lz / 2.0, 0.0, float(fy_mag), 0.0))
    else:
        cpx, cpy, cpz = st.columns(3)
        lp_x = cpx.number_input("Pos X", 0.0, float(Lx), float(Lx), float(dx))
        lp_y = cpy.number_input("Pos Y", 0.0, float(Ly), 0.0, float(dy))
        lp_z = cpz.number_input("Pos Z", 0.0, float(Lz), float(Lz/2.0), float(dz))

        cfx, cfy, cfz = st.columns(3)
        lf_x = cfx.number_input("Fx", value=0.0, step=10.0)
        lf_y = cfy.number_input("Fy", value=-100.0, step=10.0)
        lf_z = cfz.number_input("Fz", value=0.0, step=10.0)
        applied_loads.append((lp_x, lp_y, lp_z, float(lf_x), float(lf_y), float(lf_z)))

    st.markdown('<div class="sec-head">4. Parametri</div>', unsafe_allow_html=True)
    volfrac = st.slider("Frazione volume (Vf)", 0.05, 0.80, 0.20, 0.05)
    alpha = st.slider("Peso buckling (α)", 0.00, 0.90, 0.30, 0.05)
    rmin = st.slider("Filtro r_min (mm)", 0.5, 6.0, 1.2, 0.1)
    max_iter = st.slider("Iterazioni", 5, 80, 25, 5)
    solver_opt = st.selectbox("Solutore", ["PCG + Jacobi", "Diretto"])
    solver_key = "pcg" if "PCG" in solver_opt else "direct"

    st.write("")
    run_btn = st.button("RUN OPTIMIZATION", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# EXECUTION (RUNS ON BUTTON CLICK, THEN RENDERS RESULTS)
# -----------------------------------------------------------------------------
progress_holder = st.empty()
status_holder = st.empty()

if run_btn:
    progress_bar = progress_holder.progress(0)
    status_holder.text("Inizializzazione elementi H8...")

    opt = SIMPOptimizer3D(
        nelx=int(nelx), nely=int(nely), nelz=int(nelz),
        dx=float(dx), dy=float(dy), dz=float(dz),
        E0=1.0, Emin=1e-9, nu=0.3,
        penal=3.0, penal_g=6.0,
        rmin=float(rmin), volfrac=float(volfrac),
        solver_type=solver_key
    )

    # Boundaries
    if support_mode == "Incastro x=0":
        opt.fix_face("left", fix_x=True, fix_y=True, fix_z=True)
    elif support_mode == "Appoggio base":
        opt.fix_node(0, 0, 0, fix_x=True, fix_y=True, fix_z=True)
        opt.fix_node(0, 0, nelz, fix_x=True, fix_y=True, fix_z=True)
        opt.fix_node(nelx, 0, 0, fix_x=False, fix_y=True, fix_z=True)
        opt.fix_node(nelx, 0, nelz, fix_x=False, fix_y=True, fix_z=True)
    else:
        node_i = int(np.clip(round(sup_x / dx), 0, nelx))
        node_j = int(np.clip(round(sup_y / dy), 0, nely))
        node_k = int(np.clip(round(sup_z / dz), 0, nelz))
        opt.fix_node(node_i, node_j, node_k, fix_x=fix_u, fix_y=fix_v, fix_z=fix_w)

    # Loads
    for lx, ly, lz, fx, fy, fz in applied_loads:
        node_i = int(np.clip(round(lx / dx), 0, nelx))
        node_j = int(np.clip(round(ly / dy), 0, nely))
        node_k = int(np.clip(round(lz / dz), 0, nelz))
        opt.add_load(node_i, node_j, node_k, fx=fx, fy=fy, fz=fz)

    def on_progress(it, max_it, comp=0.0, vol=0.0, *args):
        pct = int((it / max_it) * 100)
        progress_bar.progress(pct)
        status_holder.text(f"Iterazione {it:02d}/{max_it:02d} | Compliance: {comp:.3e} | Vf: {vol*100:.1f}%")

    mode = "buckling_max" if alpha > 0.0 else "compliance"
    start_time = time.time()
    res = opt.solve(
        max_iter=int(max_iter),
        tol=0.015,
        mode=mode,
        alpha_buckling=float(alpha),
        progress_callback=on_progress
    )
    elapsed = time.time() - start_time

    st.session_state["res3d"] = res
    st.session_state["opt3d"] = opt
    progress_holder.empty()
    status_holder.text(f"Ottimizzazione completata in {res.iterations_run} iterazioni ({elapsed:.1f}s, RAM: {res.peak_memory_mb:.1f}MB)")


# -----------------------------------------------------------------------------
# MAIN VIEWPORT: HEADER & TOOLBAR
# -----------------------------------------------------------------------------
has_result = ("res3d" in st.session_state and st.session_state["res3d"] is not None)

view_choice = "preview"
threshold_val = 0.35
smooth_val = 6
selected_stress_key = "von_mises"
selected_stress_cmap = "Turbo"
selected_stress_name = "Von Mises"

if not has_result:
    st.markdown(f'<div style="font-size:0.85rem; color:#71717a; margin-bottom:0.5rem; letter-spacing:0.05em;">DOMINIO 3D: {Lx:.1f} × {Ly:.1f} × {Lz:.1f} mm | {nelx}×{nely}×{nelz} elementi ({nelx*nely*nelz:,} voxel)</div>', unsafe_allow_html=True)
else:
    # Full Result Toolbar
    col_v1, col_v2, col_v3, col_v4 = st.columns([3, 2, 2, 1])
    view_choice = col_v1.radio(
        "Modalità Vista:",
        ["tension_compression", "cad", "stress", "preview"],
        index=0,
        format_func=lambda x: "Trazione / Compressione" if x=="tension_compression" else ("Monocromatico CAD" if x=="cad" else ("Sforzi Termici" if x=="stress" else "Solo Dominio")),
        horizontal=True,
        label_visibility="collapsed"
    )
    threshold_val = col_v2.slider("Soglia densità", 0.05, 0.90, 0.30, 0.05)
    smooth_val = col_v3.slider("Smoothing (Taubin)", 0, 15, 6, 1)
    if col_v4.button("RESET", use_container_width=True):
        del st.session_state["res3d"]
        if "opt3d" in st.session_state:
            del st.session_state["opt3d"]
        st.rerun()

    if view_choice == "stress":
        sc1, sc2 = st.columns(2)
        stress_label = sc1.selectbox(
            "Componente Sforzo:",
            [
                "Von Mises",
                "σ_III Minimo (Compressione)",
                "σ_I Massimo (Trazione)",
                "σ_xx", "σ_yy", "σ_zz"
            ]
        )
        key_map = {
            "Von Mises": ("von_mises", "Von Mises"),
            "σ_III Minimo (Compressione)": ("sigma_III", "σ_III"),
            "σ_I Massimo (Trazione)": ("sigma_I", "σ_I"),
            "σ_xx": ("sigma_xx", "σ_xx"),
            "σ_yy": ("sigma_yy", "σ_yy"),
            "σ_zz": ("sigma_zz", "σ_zz")
        }
        selected_stress_key, selected_stress_name = key_map.get(stress_label, ("von_mises", "Von Mises"))
        selected_stress_cmap = sc2.selectbox("Colormap:", ["Turbo", "Jet", "Inferno", "Plasma", "Hot"], index=0)

# Render 3D Domain Figure (Plotly ModeBar hidden for clean view)
preview_fig = build_domain_preview_figure(
    nelx=nelx, nely=nely, nelz=nelz,
    dx=dx, dy=dy, dz=dz,
    fixed_nodes_coords=custom_supports,
    applied_loads=applied_loads,
    res=st.session_state.get("res3d", None),
    threshold=threshold_val,
    view_mode=view_choice if has_result else "preview",
    smooth_iterations=smooth_val,
    stress_field_key=selected_stress_key,
    stress_colormap=selected_stress_cmap,
    stress_field_name=selected_stress_name
)
st.plotly_chart(preview_fig, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# POST-PROCESSING: METRICS, STRESSES, STL EXPORT
# -----------------------------------------------------------------------------
if has_result:
    res = st.session_state["res3d"]
    opt = st.session_state["opt3d"]

    st.markdown('<div class="sec-head">Risultati Ottimizzazione</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Compliance", f"{res.compliance:.3e} N·mm")
    blf_str = f"{res.blf_history[-1]:.4f}" if res.blf_history else "N/A"
    m2.metric("BLF Instabilità", blf_str)
    m3.metric("Frazione Volume", f"{res.volume_fraction*100:.1f}%")
    m4.metric("Tempo Calcolo", f"{res.execution_time_sec:.1f}s")

    if res.stresses is not None:
        s1, s2, s3 = st.columns(3)
        vm = res.stresses.get("von_mises", np.array([0.0]))
        s3_min = res.stresses.get("sigma_III", np.array([0.0]))
        s1_max = res.stresses.get("sigma_I", np.array([0.0]))
        s1.metric("Max Von Mises", f"{float(np.max(vm)):.2f} MPa")
        s2.metric("Picco Compressione (σ_III)", f"{float(np.min(s3_min)):.2f} MPa")
        s3.metric("Picco Trazione (σ_I)", f"{float(np.max(s1_max)):.2f} MPa")

    col_chart, col_stl = st.columns([1, 1])

    with col_chart:
        if res.compliance_history:
            fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
            iters = list(range(1, len(res.compliance_history) + 1))
            fig_hist.add_trace(
                go.Scatter(x=iters, y=res.compliance_history, name="Compliance", line=dict(color="#ffffff", width=1.5)),
                secondary_y=False
            )
            if res.blf_history:
                fig_hist.add_trace(
                    go.Scatter(x=iters, y=res.blf_history, name="BLF", line=dict(color="#00f5d4", width=1.5, dash="dot")),
                    secondary_y=True
                )
            fig_hist.update_layout(
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                height=180,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.2, font=dict(family="Inconsolata", color="#71717a", size=9)),
                xaxis=dict(gridcolor="#18181b", color="#52525b"),
                yaxis=dict(gridcolor="#18181b", color="#52525b"),
                yaxis2=dict(gridcolor="#18181b", color="#52525b")
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with col_stl:
        st.markdown('<div class="sec-head">Esportazione STL (Watertight Manifold)</div>', unsafe_allow_html=True)
        stl_t = st.slider("Soglia densità STL", 0.05, 0.90, 0.30, 0.05, key="stl_thresh")

        c_down1, c_down2 = st.columns(2)
        with c_down1:
            if st.button("GENERA STL LISCIO", use_container_width=True):
                tmp_stl = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                tmp_stl.close()
                opt.export_smooth_stl(res, filepath=tmp_stl.name, threshold=stl_t, smooth_iterations=10)
                with open(tmp_stl.name, "rb") as f:
                    stl_bytes = f.read()
                st.download_button(
                    label="DOWNLOAD STL LISCIO",
                    data=stl_bytes,
                    file_name=f"CE_3D_Smooth_{nelx}x{nely}x{nelz}.stl",
                    mime="application/sla",
                    use_container_width=True
                )

        with c_down2:
            if st.button("GENERA STL VOXEL", use_container_width=True):
                tmp_stl = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                tmp_stl.close()
                opt.export_stl(res, filepath=tmp_stl.name, threshold=stl_t)
                with open(tmp_stl.name, "rb") as f:
                    stl_bytes = f.read()
                st.download_button(
                    label="DOWNLOAD STL RAW",
                    data=stl_bytes,
                    file_name=f"CE_3D_Raw_{nelx}x{nely}x{nelz}.stl",
                    mime="application/sla",
                    use_container_width=True
                )

    pdf_path = PROJECT_ROOT / "docs" / "TO_3D_log.pdf"
    if pdf_path.exists():
        with st.expander("TO_3D_log.pdf"):
            with open(pdf_path, "rb") as f:
                st.download_button("DOWNLOAD PDF", data=f.read(), file_name="TO_3D_log.pdf", mime="application/pdf")
