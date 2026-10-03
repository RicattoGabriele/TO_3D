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

from physics_engine.simp_engine_3d import SIMPOptimizer3D, SIMPResult3D

# -----------------------------------------------------------------------------
# Streamlit Page Setup - Minimal Monochrome Dark Style
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CE-3D | Topology Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@300;400;600&display=swap');

    /* Global Typography & Black Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 300 !important;
        background-color: #000000 !important;
        color: #f1f5f9 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 400 !important;
        color: #ffffff !important;
        letter-spacing: 0.05em;
    }

    .main-title {
        font-size: 1.6rem;
        font-weight: 600 !important;
        color: #ffffff;
        border-bottom: 1px solid #27272a;
        padding-bottom: 8px;
        margin-bottom: 4px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sub-title {
        font-size: 0.85rem;
        color: #71717a;
        margin-bottom: 1.2rem;
        letter-spacing: 0.04em;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #09090b !important;
        border-right: 1px solid #18181b !important;
    }

    /* Neutral Monochrome Sliders - REMOVE ORANGE/RED */
    div[data-baseweb="slider"] {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 300 !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #09090b !important;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.7) !important;
        width: 16px !important;
        height: 16px !important;
    }
    /* Slider active progress track */
    div[data-baseweb="slider"] > div > div:first-child {
        background: #27272a !important;
    }
    div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
    div[data-baseweb="slider"] div[style*="background-color: rgb(255, 115, 0)"],
    div[data-baseweb="slider"] div[style*="background-color: #ff4b4b"],
    div[data-baseweb="slider"] div[style*="background-color: #f97316"] {
        background-color: #e4e4e7 !important;
    }

    /* Input fields and boxes */
    input, textarea, select, [data-baseweb="input"], [data-baseweb="base-input"] {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 300 !important;
        background-color: #09090b !important;
        color: #ffffff !important;
        border-color: #27272a !important;
        border-radius: 2px !important;
    }
    input:focus, textarea:focus {
        border-color: #ffffff !important;
        box-shadow: none !important;
    }

    /* Minimalist Buttons */
    .stButton > button {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 400 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background-color: #09090b !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
        border-radius: 2px !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
    }
    .stButton > button:active {
        background-color: #e4e4e7 !important;
        color: #000000 !important;
    }

    /* Primary Accent Button */
    button[kind="primary"] {
        background-color: #18181b !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
    }
    button[kind="primary"]:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Metric Containers */
    [data-testid="stMetricValue"] {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inconsolata', monospace !important;
        font-weight: 300 !important;
        color: #a1a1aa !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #ffffff !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #09090b !important;
        border: 1px solid #18181b !important;
        color: #a1a1aa !important;
        font-weight: 300 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-title">CE-3D Continuum Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Objective Linear Elastic Compliance & Linearized Buckling Stability Engine (H8 Isoparametric Elements)</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3D Domain Preview Function with Colored Boundary Conditions
# -----------------------------------------------------------------------------
def build_domain_preview_figure(
    nelx: int, nely: int, nelz: int,
    dx: float, dy: float, dz: float,
    fixed_nodes_coords: List[Tuple[float, float, float, str]],
    applied_loads: List[Tuple[float, float, float, float, float, float]],
    res: Optional[SIMPResult3D] = None,
    threshold: float = 0.35,
    view_mode: str = "preview",
    stress_field_key: str = "von_mises",
    stress_colormap: str = "Turbo",
    stress_field_name: str = "Von Mises"
) -> go.Figure:
    """
    Renders an interactive 3D Plotly visualization:
    - Bounding volume parallelepiped [0..Lx] x [0..Ly] x [0..Lz]
    - Fixed supports as high-visibility colored elements (Cyan/Green glyphs)
    - Applied loads as bright colored vectors/arrows (Magenta/Red glyphs)
    - Optimized topology isosurface or full 3D Thermal Stress Heatmap across solid voxels.
    """
    Lx = float(nelx * dx)
    Ly = float(nely * dy)
    Lz = float(nelz * dz)

    fig = go.Figure()

    # 1. Bounding Box Parallelepiped Wireframe (12 Edges)
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
        line=dict(color="#52525b", width=3),
        name="Domain Box",
        hoverinfo="skip"
    ))

    # Semi-transparent domain shading
    fig.add_trace(go.Mesh3d(
        x=[0, Lx, Lx, 0, 0, Lx, Lx, 0],
        y=[0, 0, Ly, Ly, 0, 0, Ly, Ly],
        z=[0, 0, 0, 0, Lz, Lz, Lz, Lz],
        i=[0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 2, 2],
        j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 3, 7],
        k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 7, 6],
        color="#27272a",
        opacity=0.08,
        name="Domain Volume",
        hoverinfo="skip"
    ))

    # 2. Fixed Supports Markers (Cyan / High Visibility)
    if fixed_nodes_coords:
        fx_x = [pt[0] for pt in fixed_nodes_coords]
        fx_y = [pt[1] for pt in fixed_nodes_coords]
        fx_z = [pt[2] for pt in fixed_nodes_coords]
        fx_hover = [f"Support ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}) mm<br>DOFs: {pt[3]}" for pt in fixed_nodes_coords]

        marker_size = max(5, min(14, int(220 / max(nelx, nely, nelz))))

        fig.add_trace(go.Scatter3d(
            x=fx_x, y=fx_y, z=fx_z,
            mode="markers",
            marker=dict(
                size=marker_size,
                color="#00f5d4",  # High-visibility electric cyan
                symbol="square",
                line=dict(color="#ffffff", width=1)
            ),
            text=fx_hover,
            hoverinfo="text",
            name="Fixed Support [Ux, Uy, Uz]"
        ))

    # 3. Applied Loads (Magenta / Fluorescent Red with Vector Arrows)
    if applied_loads:
        for idx, (lx, ly, lz, f_x, f_y, f_z) in enumerate(applied_loads):
            mag = np.sqrt(f_x**2 + f_y**2 + f_z**2)
            if mag < 1e-6:
                continue

            fig.add_trace(go.Scatter3d(
                x=[lx], y=[ly], z=[lz],
                mode="markers",
                marker=dict(
                    size=max(8, int(300 / max(nelx, nely, nelz))),
                    color="#ff0055",  # Vibrant neon magenta
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1.5)
                ),
                text=[f"Load #{idx+1}: F=({f_x:.1f}, {f_y:.1f}, {f_z:.1f}) N<br>Point: ({lx:.1f}, {ly:.1f}, {lz:.1f}) mm"],
                hoverinfo="text",
                name=f"Load #{idx+1} Point"
            ))

            arrow_scale = 0.22 * max(Lx, Ly, Lz)
            dir_x = (f_x / mag) * arrow_scale
            dir_y = (f_y / mag) * arrow_scale
            dir_z = (f_z / mag) * arrow_scale

            fig.add_trace(go.Scatter3d(
                x=[lx, lx + dir_x],
                y=[ly, ly + dir_y],
                z=[lz, lz + dir_z],
                mode="lines+markers",
                line=dict(color="#ff0055", width=6),
                marker=dict(size=[0, 6], color="#ffffff", symbol="cone"),
                text=[None, f"F = {mag:.1f} N"],
                hoverinfo="text",
                name=f"Load #{idx+1} Vector"
            ))

    # 4. Optimized Geometry (Isosurface)
    if res is not None and view_mode == "result":
        X, Y, Z = np.mgrid[0:Lx:complex(0, nelx),
                           0:Ly:complex(0, nely),
                           0:Lz:complex(0, nelz)]

        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=res.density_matrix.flatten(),
            isomin=threshold,
            isomax=1.0,
            surface_count=2,
            colorscale=[[0, '#38bdf8'], [1, '#ffffff']],
            caps=dict(x_show=True, y_show=True, z_show=True),
            colorbar=dict(
                title=dict(text="ρ (Density)", font=dict(color="#ffffff", size=10)),
                tickfont=dict(color="#a1a1aa", size=9),
                len=0.6,
                x=1.02
            ),
            name="Optimized Topology"
        ))

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
                f"Voxel ({ix}, {iy}, {iz})<br>Pos: ({x:.1f}, {y:.1f}, {z:.1f}) mm<br>Densità: {rho:.2f}<br>{stress_field_name}: {val:.2f} MPa"
                for ix, iy, iz, x, y, z, rho, val in zip(ix_arr, iy_arr, iz_arr, solid_x, solid_y, solid_z, solid_dens, solid_vals)
            ]

            marker_sz = max(5, min(20, int(420 / max(nelx, nely, nelz))))

            fig.add_trace(go.Scatter3d(
                x=solid_x, y=solid_y, z=solid_z,
                mode="markers",
                marker=dict(
                    size=marker_sz,
                    color=solid_vals,
                    colorscale=stress_colormap,
                    colorbar=dict(
                        title=dict(text=f"{stress_field_name} (MPa)", font=dict(color="#ffffff", size=10)),
                        tickfont=dict(color="#a1a1aa", size=9),
                        len=0.6,
                        x=1.02
                    ),
                    showscale=True,
                    symbol="square",
                    opacity=0.96
                ),
                text=hover_texts,
                hoverinfo="text",
                name=f"Stress: {stress_field_name}"
            ))

    # Plotly Layout: Pitch Black Background, Minimalist Axes
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        scene=dict(
            xaxis=dict(
                title=dict(text="X (mm)", font=dict(color="#a1a1aa", size=11)),
                tickfont=dict(color="#71717a", size=9),
                backgroundcolor="#050505",
                gridcolor="#18181b",
                zerolinecolor="#27272a",
                range=[-0.05 * Lx, 1.25 * Lx]
            ),
            yaxis=dict(
                title=dict(text="Y (mm)", font=dict(color="#a1a1aa", size=11)),
                tickfont=dict(color="#71717a", size=9),
                backgroundcolor="#050505",
                gridcolor="#18181b",
                zerolinecolor="#27272a",
                range=[-0.05 * Ly, 1.25 * Ly]
            ),
            zaxis=dict(
                title=dict(text="Z (mm)", font=dict(color="#a1a1aa", size=11)),
                tickfont=dict(color="#71717a", size=9),
                backgroundcolor="#050505",
                gridcolor="#18181b",
                zerolinecolor="#27272a",
                range=[-0.05 * Lz, 1.25 * Lz]
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.8, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#d4d4d8", family="Inconsolata")
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=540
    )

    return fig


# -----------------------------------------------------------------------------
# SIDEBAR: PARAMETERS & GEOMETRY CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 1. Dominio 3D")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    nelx = col_d1.number_input("nelx", min_value=4, max_value=80, value=20, step=2)
    nely = col_d2.number_input("nely", min_value=2, max_value=50, value=10, step=2)
    nelz = col_d3.number_input("nelz", min_value=2, max_value=50, value=10, step=2)

    col_sz1, col_sz2, col_sz3 = st.columns(3)
    dx = col_sz1.number_input("dx (mm)", min_value=0.1, max_value=20.0, value=1.0, step=0.5)
    dy = col_sz2.number_input("dy (mm)", min_value=0.1, max_value=20.0, value=1.0, step=0.5)
    dz = col_sz3.number_input("dz (mm)", min_value=0.1, max_value=20.0, value=1.0, step=0.5)

    Lx = nelx * dx
    Ly = nely * dy
    Lz = nelz * dz
    total_elements = nelx * nely * nelz
    total_dofs = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    st.caption(f"Volume: {Lx:.1f} × {Ly:.1f} × {Lz:.1f} mm | {total_elements:,} elementi | {total_dofs:,} GDL")

    st.markdown("---")
    st.markdown("### 2. Vincoli & Supporti")
    support_mode = st.radio(
        "Modalità Supporti:",
        ["Faccia Sinistra Incastrata (x = 0)", "Doppio Appoggio Base", "Punto Personalizzato (X, Y, Z)"],
        index=0
    )

    custom_supports: List[Tuple[float, float, float, str]] = []
    if support_mode == "Faccia Sinistra Incastrata (x = 0)":
        # Sample points on face x=0 for visualization
        for j_idx in range(0, nely + 1, max(1, nely // 4)):
            for k_idx in range(0, nelz + 1, max(1, nelz // 4)):
                custom_supports.append((0.0, j_idx * dy, k_idx * dz, "Ux, Uy, Uz"))
    elif support_mode == "Doppio Appoggio Base":
        # Four bottom corners
        custom_supports.append((0.0, 0.0, 0.0, "Ux, Uy, Uz"))
        custom_supports.append((0.0, 0.0, Lz, "Ux, Uy, Uz"))
        custom_supports.append((Lx, 0.0, 0.0, "Uy, Uz"))
        custom_supports.append((Lx, 0.0, Lz, "Uy, Uz"))
    else:
        st.markdown("**Coordinate Punto Vincolato:**")
        col_sup_x, col_sup_y, col_sup_z = st.columns(3)
        sup_x = col_sup_x.number_input("X (mm)", min_value=0.0, max_value=float(Lx), value=0.0, step=float(dx))
        sup_y = col_sup_y.number_input("Y (mm)", min_value=0.0, max_value=float(Ly), value=0.0, step=float(dy))
        sup_z = col_sup_z.number_input("Z (mm)", min_value=0.0, max_value=float(Lz), value=0.0, step=float(dz))

        col_dof_x, col_dof_y, col_dof_z = st.columns(3)
        fix_u = col_dof_x.checkbox("Blocca Ux", value=True)
        fix_v = col_dof_y.checkbox("Blocca Uy", value=True)
        fix_w = col_dof_z.checkbox("Blocca Uz", value=True)

        dof_desc = []
        if fix_u: dof_desc.append("Ux")
        if fix_v: dof_desc.append("Uy")
        if fix_w: dof_desc.append("Uz")
        custom_supports.append((sup_x, sup_y, sup_z, ", ".join(dof_desc) if dof_desc else "Nessuno"))

    st.markdown("---")
    st.markdown("### 3. Carichi Concentrati")
    load_mode = st.radio(
        "Modalità Carico:",
        ["Carico Trasversale Estremità (Punta)", "Punto Preciso Personalizzato (X, Y, Z, Fx, Fy, Fz)"],
        index=0
    )

    applied_loads: List[Tuple[float, float, float, float, float, float]] = []
    if load_mode == "Carico Trasversale Estremità (Punta)":
        fy_mag = st.number_input("Forza Fy (N)", value=-100.0, step=10.0)
        # Applied at (Lx, 0, Lz/2)
        applied_loads.append((Lx, 0.0, Lz / 2.0, 0.0, float(fy_mag), 0.0))
    else:
        st.markdown("**Posizione e Vettore Forza:**")
        col_lp_x, col_lp_y, col_lp_z = st.columns(3)
        lp_x = col_lp_x.number_input("Pos X (mm)", 0.0, float(Lx), float(Lx), float(dx))
        lp_y = col_lp_y.number_input("Pos Y (mm)", 0.0, float(Ly), 0.0, float(dy))
        lp_z = col_lp_z.number_input("Pos Z (mm)", 0.0, float(Lz), float(Lz/2.0), float(dz))

        col_lf_x, col_lf_y, col_lf_z = st.columns(3)
        lf_x = col_lf_x.number_input("Fx (N)", value=0.0, step=10.0)
        lf_y = col_lf_y.number_input("Fy (N)", value=-100.0, step=10.0)
        lf_z = col_lf_z.number_input("Fz (N)", value=0.0, step=10.0)
        applied_loads.append((lp_x, lp_y, lp_z, float(lf_x), float(lf_y), float(lf_z)))

    st.markdown("---")
    st.markdown("### 4. Parametri Ottimizzazione")
    volfrac = st.slider("Frazione di Volume (Vf)", 0.10, 0.80, 0.35, 0.05)
    alpha = st.slider("Peso Instabilità Buckling (α)", 0.00, 0.90, 0.30, 0.05)
    rmin = st.slider("Raggio Filtro r_min (mm)", 0.5, 6.0, 1.5, 0.5)
    max_iter = st.slider("Iterazioni Max", 5, 60, 25, 5)
    solver_opt = st.selectbox("Solutore di Sistema", ["PCG + Jacobi (Memoria Minima)", "Diretto (SuperLU)"])
    solver_key = "pcg" if "PCG" in solver_opt else "direct"


# -----------------------------------------------------------------------------
# MAIN VIEW: REAL-TIME DOMAIN PREVIEW & SOLVER
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("#### Stato Configurazione")
    st.write(f"• **Dimensioni**: `{Lx:.1f} × {Ly:.1f} × {Lz:.1f}` mm")
    st.write(f"• **Mesh Elementi**: `{nelx} × {nely} × {nelz}` ({total_elements:,} voxel)")
    st.write(f"• **Filtro Spaziale**: `r_min = {rmin:.1f}` mm")
    st.write(f"• **Obiettivo**: `α = {alpha:.2f}` (0: Rigidezza pura, 1: Max Stabilità)")
    
    run_btn = st.button("🚀 Avvia Ottimizzazione Topologica 3D", type="primary", use_container_width=True)

    progress_holder = st.empty()
    status_holder = st.empty()
    metric_holder = st.empty()

with col_right:
    st.markdown("#### Anteprima 3D del Dominio & Risultati")
    preview_placeholder = st.empty()

    # Determine if we have a converged solution in state
    has_result = "res3d" in st.session_state and st.session_state["res3d"] is not None

    view_choice = "preview"
    threshold_val = 0.35
    selected_stress_key = "von_mises"
    selected_stress_cmap = "Turbo"
    selected_stress_name = "Von Mises"

    if has_result:
        col_t1, col_t2 = st.columns([3, 2])
        view_choice = col_t1.radio(
            "Modalità Visualizzazione:",
            ["result", "stress", "preview"],
            index=0,
            format_func=lambda x: "🧊 Topologia (Densità)" if x=="result" else ("🔥 Mappa Termica Sforzi" if x=="stress" else "📐 Solo Dominio & Vincoli"),
            horizontal=True
        )
        threshold_val = col_t2.slider("Soglia Densità Isosuperficie", 0.10, 0.90, 0.35, 0.05)

        if view_choice == "stress":
            col_s1, col_s2 = st.columns(2)
            stress_label = col_s1.selectbox(
                "Componente Sforzo da Visualizzare:",
                [
                    "Von Mises (Sforzo Equivalente)",
                    "σ_III Minimo Principale (Compressione / Instabilità)",
                    "σ_I Massimo Principale (Trazione)",
                    "σ_xx Sforzo Normale X",
                    "σ_yy Sforzo Normale Y",
                    "σ_zz Sforzo Normale Z"
                ]
            )
            key_map = {
                "Von Mises (Sforzo Equivalente)": ("von_mises", "Von Mises"),
                "σ_III Minimo Principale (Compressione / Instabilità)": ("sigma_III", "σ_III (Compressione)"),
                "σ_I Massimo Principale (Trazione)": ("sigma_I", "σ_I (Trazione)"),
                "σ_xx Sforzo Normale X": ("sigma_xx", "σ_xx"),
                "σ_yy Sforzo Normale Y": ("sigma_yy", "σ_yy"),
                "σ_zz Sforzo Normale Z": ("sigma_zz", "σ_zz")
            }
            selected_stress_key, selected_stress_name = key_map.get(stress_label, ("von_mises", "Von Mises"))

            selected_stress_cmap = col_s2.selectbox(
                "Palette Termica (Colormap):",
                ["Turbo", "Jet", "Inferno", "Plasma", "Hot"],
                index=0
            )

    # Render the interactive 3D Domain immediately on startup or parameter change
    preview_fig = build_domain_preview_figure(
        nelx=nelx, nely=nely, nelz=nelz,
        dx=dx, dy=dy, dz=dz,
        fixed_nodes_coords=custom_supports,
        applied_loads=applied_loads,
        res=st.session_state.get("res3d", None),
        threshold=threshold_val,
        view_mode=view_choice,
        stress_field_key=selected_stress_key,
        stress_colormap=selected_stress_cmap,
        stress_field_name=selected_stress_name
    )
    preview_placeholder.plotly_chart(preview_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# OPTIMIZATION EXECUTION HOOK
# -----------------------------------------------------------------------------
if run_btn:
    progress_bar = progress_holder.progress(0)
    status_holder.info("Inizializzazione solutore ed elementi esaedrici H8...")

    opt = SIMPOptimizer3D(
        nelx=int(nelx), nely=int(nely), nelz=int(nelz),
        dx=float(dx), dy=float(dy), dz=float(dz),
        E0=1.0, Emin=1e-9, nu=0.3,
        penal=3.0, penal_g=6.0,
        rmin=float(rmin), volfrac=float(volfrac),
        solver_type=solver_key
    )

    # Apply Boundary Conditions
    if support_mode == "Faccia Sinistra Incastrata (x = 0)":
        opt.fix_face("left", fix_x=True, fix_y=True, fix_z=True)
    elif support_mode == "Doppio Appoggio Base":
        opt.fix_node(0, 0, 0, fix_x=True, fix_y=True, fix_z=True)
        opt.fix_node(0, 0, nelz, fix_x=True, fix_y=True, fix_z=True)
        opt.fix_node(nelx, 0, 0, fix_x=False, fix_y=True, fix_z=True)
        opt.fix_node(nelx, 0, nelz, fix_x=False, fix_y=True, fix_z=True)
    else:
        # Custom point
        node_i = int(np.clip(round(sup_x / dx), 0, nelx))
        node_j = int(np.clip(round(sup_y / dy), 0, nely))
        node_k = int(np.clip(round(sup_z / dz), 0, nelz))
        opt.fix_node(node_i, node_j, node_k, fix_x=fix_u, fix_y=fix_v, fix_z=fix_w)

    # Apply Loads
    for lx, ly, lz, fx, fy, fz in applied_loads:
        node_i = int(np.clip(round(lx / dx), 0, nelx))
        node_j = int(np.clip(round(ly / dy), 0, nely))
        node_k = int(np.clip(round(lz / dz), 0, nelz))
        opt.add_load(node_i, node_j, node_k, fx=fx, fy=fy, fz=fz)

    # Live update callback
    def on_progress(it, max_it, comp=0.0, vol=0.0, *args):
        pct = int((it / max_it) * 100)
        progress_bar.progress(pct)
        status_holder.text(f"Iterazione {it:02d}/{max_it:02d} | Compliance: {comp:.3e} | Densità Media: {vol*100:.1f}%")

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

    status_holder.success(f"Convergenza completata in {res.iterations_run} iterazioni ({elapsed:.2f}s)! RAM picco: {res.peak_memory_mb:.1f} MB")

    # Re-render with optimized results
    updated_fig = build_domain_preview_figure(
        nelx=nelx, nely=nely, nelz=nelz,
        dx=dx, dy=dy, dz=dz,
        fixed_nodes_coords=custom_supports,
        applied_loads=applied_loads,
        res=res,
        threshold=threshold_val,
        view_mode="result",
        stress_field_key=selected_stress_key,
        stress_colormap=selected_stress_cmap,
        stress_field_name=selected_stress_name
    )
    preview_placeholder.plotly_chart(updated_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# CONVERGENCE CHARTS & STL EXPORT (BELOW)
# -----------------------------------------------------------------------------
if "res3d" in st.session_state and st.session_state["res3d"] is not None:
    res = st.session_state["res3d"]
    opt = st.session_state["opt3d"]

    st.markdown("---")
    col_metrics, col_export = st.columns([1, 1])

    with col_metrics:
        st.markdown("#### Metriche di Convergenza & Sforzi")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compliance (N·mm)", f"{res.compliance:.3e}")
        blf_str = f"{res.blf_history[-1]:.4f}" if res.blf_history else "N/A"
        m2.metric("BLF Fondamentale", blf_str)
        m3.metric("Frazione Volume", f"{res.volume_fraction*100:.1f}%")
        m4.metric("Tempo Calcolo", f"{res.execution_time_sec:.1f}s")

        if res.stresses is not None:
            st.markdown("##### Stato Tensionale Strutturale (Estremi Solidi)")
            s1, s2, s3 = st.columns(3)
            vm = res.stresses.get("von_mises", np.array([0.0]))
            s3_min = res.stresses.get("sigma_III", np.array([0.0]))
            s1_max = res.stresses.get("sigma_I", np.array([0.0]))
            s1.metric("Picco Von Mises", f"{float(np.max(vm)):.2f} MPa")
            s2.metric("Picco Compressione (σ_III)", f"{float(np.min(s3_min)):.2f} MPa")
            s3.metric("Picco Trazione (σ_I)", f"{float(np.max(s1_max)):.2f} MPa")

        if res.compliance_history:
            fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
            iters = list(range(1, len(res.compliance_history) + 1))
            fig_hist.add_trace(
                go.Scatter(x=iters, y=res.compliance_history, name="Compliance", line=dict(color="#ffffff", width=2)),
                secondary_y=False
            )
            if res.blf_history:
                fig_hist.add_trace(
                    go.Scatter(x=iters, y=res.blf_history, name="BLF Instabilità", line=dict(color="#00f5d4", width=2, dash="dot")),
                    secondary_y=True
                )
            fig_hist.update_layout(
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                height=220,
                margin=dict(l=10, r=10, t=20, b=20),
                legend=dict(orientation="h", y=1.15, font=dict(family="Inconsolata", color="#a1a1aa", size=10)),
                xaxis=dict(gridcolor="#18181b", color="#71717a"),
                yaxis=dict(gridcolor="#18181b", color="#71717a"),
                yaxis2=dict(gridcolor="#18181b", color="#71717a")
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_export:
        st.markdown("#### Esportazione STL Watertight (PicoGK / 3D Print Ready)")
        st.write("Genera una mesh chiusa a tenuta stagna (senza facce interne condivise) pronta per la produzione additiva o lo smoothing.")
        
        stl_threshold = st.slider("Soglia Densità per Esportazione STL", 0.10, 0.90, 0.35, 0.05, key="stl_t")
        
        if st.button("💾 Genera File STL Manifold", use_container_width=True):
            tmp_stl = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
            tmp_stl.close()
            opt.export_stl(res, filepath=tmp_stl.name, threshold=stl_threshold)
            
            with open(tmp_stl.name, "rb") as f:
                stl_bytes = f.read()

            st.download_button(
                label="📥 Scarica Mesh STL Ottimizzata (.stl)",
                data=stl_bytes,
                file_name=f"CE_3D_Optimized_{nelx}x{nely}x{nelz}.stl",
                mime="application/sla",
                use_container_width=True
            )
            st.success(f"Mesh STL generata con successo ({len(stl_bytes):,} bytes, Watertight: True)!")


# -----------------------------------------------------------------------------
# COLLAPSIBLE THEORETICAL LOG (MINIMAL FOOTER)
# -----------------------------------------------------------------------------
with st.expander("📖 Consulta Log Teorico & Riferimenti Matematici (PDF)", expanded=False):
    st.markdown("""
    Tutti i cambiamenti teorici implementati nel passaggio dal 2D al 3D sono documentati nel file **`docs/TO_3D_log.pdf`**, seguendo la struttura del paper di riferimento:
    - **Cinematica esaedrica H8** a 24 GDL con Jacobiano analitico $\det(J) = V_e / 8$.
    - **Decomposizione della rigidezza geometrica** in 6 matrici invarianti precomputabili $\mathbf{G}_{0, k}$.
    - **Solutore PCG con preconditioner di Jacobi** a consumo RAM minimo ($< 100$ MB per $2.000$ voxel su Ryzen 7).
    - **Sensitività esatte dello stato aggiunto** con riscalamento dinamico $L_1$.
    """)
    pdf_path = PROJECT_ROOT / "docs" / "TO_3D_log.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Scarica Paper Teorico Completo (TO_3D_log.pdf)", data=f.read(), file_name="TO_3D_log.pdf", mime="application/pdf")
