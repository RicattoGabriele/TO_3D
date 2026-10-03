"""
PicoGK Aerospace Component Generator - 3D Visualizer & Topology Metrics.
Provides interactive Plotly WebGL 3D rendering (Mesh3d) and Trimesh-based geometric
and topology metrics analysis (bounding box, volume, surface area, manifold/watertight check).
"""

import math
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Optional Plotly and Trimesh imports with robust fallbacks
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None


class FallbackMesh:
    """Lightweight pure-Python polygon mesh representation used when Trimesh is absent."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)

    @property
    def bounds(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.zeros((2, 3), dtype=np.float32)
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)], dtype=np.float32)

    @property
    def extents(self) -> np.ndarray:
        b = self.bounds
        return b[1] - b[0]

    @property
    def is_watertight(self) -> bool:
        # Simple heuristic check: each edge shared by exactly 2 faces
        if len(self.faces) == 0:
            return False
        edges = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ], axis=0)
        sorted_edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    @property
    def area(self) -> float:
        if len(self.faces) == 0:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))

    @property
    def volume(self) -> float:
        if len(self.faces) == 0 or not self.is_watertight:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        # Divergence theorem volume calculation for closed triangular mesh
        vol = np.sum(v0[:, 0] * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1]) +
                     v0[:, 1] * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2]) +
                     v0[:, 2] * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])) / 6.0
        return float(abs(vol))


def parse_binary_stl(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parses a standard binary STL file into unique vertices and triangle face indices."""
    with open(file_path, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
        if len(count_bytes) < 4:
            raise ValueError("Corrupted STL: unable to read triangle count.")
        tri_count = struct.unpack("<I", count_bytes)[0]

        # Each triangle is 50 bytes: 12 floats (48 bytes) + 2 attribute bytes
        raw_data = f.read(tri_count * 50)
        if len(raw_data) < tri_count * 50:
            # Handle possible truncated file
            tri_count = len(raw_data) // 50

    # Extract vertex coordinates from binary buffer
    # Format per triangle: normal (3f), v0 (3f), v1 (3f), v2 (3f), attr (H)
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2")
    ])
    records = np.frombuffer(raw_data[:tri_count * 50], dtype=dtype)

    all_verts = np.concatenate([
        records["v0"],
        records["v1"],
        records["v2"]
    ], axis=0)

    # Clean duplicates to generate compact index buffer
    unique_verts, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    faces = np.column_stack([
        inverse[0:tri_count],
        inverse[tri_count:2 * tri_count],
        inverse[2 * tri_count:3 * tri_count]
    ])

    return unique_verts.astype(np.float32), faces.astype(np.int32)


def parse_wavefront_obj(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parses a Wavefront OBJ file into vertices and triangular faces."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # Format can be f v1 v2 v3 or f v1/vt1/vn1 ...
                face_indices = []
                for p in parts[1:]:
                    idx_str = p.split("/")[0]
                    if idx_str:
                        idx = int(idx_str)
                        # OBJ is 1-indexed; negative index counts from end
                        idx = idx - 1 if idx > 0 else len(vertices) + idx
                        face_indices.append(idx)
                # Triangulate polygons with simple fan
                for k in range(1, len(face_indices) - 1):
                    faces.append([face_indices[0], face_indices[k], face_indices[k + 1]])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_mesh(mesh_path: str) -> Union[Any, FallbackMesh]:
    """
    Loads a 3D polygon mesh from STL or OBJ format using Trimesh if available,
    or a pure-Python fallback.
    """
    path = Path(mesh_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh file does not exist at: {path}")

    if TRIMESH_AVAILABLE:
        try:
            loaded = trimesh.load(str(path))
            if isinstance(loaded, trimesh.Scene):
                # Concatenate geometries if loaded as a Scene
                if len(loaded.geometry) > 0:
                    loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
                else:
                    raise ValueError("Empty Scene loaded from mesh file.")
            return loaded
        except Exception:
            # Fall back to custom parser if trimesh fails on specific file format
            pass

    # Pure Python Fallback
    ext = path.suffix.lower()
    if ext == ".stl":
        v, f = parse_binary_stl(str(path))
    elif ext == ".obj":
        v, f = parse_wavefront_obj(str(path))
    else:
        # Default try STL then OBJ
        try:
            v, f = parse_binary_stl(str(path))
        except Exception:
            v, f = parse_wavefront_obj(str(path))

    return FallbackMesh(v, f)


def compute_mesh_metrics(mesh_or_path: Union[str, Any]) -> Dict[str, Any]:
    """
    Calculates detailed geometric, topological, and physical metrics for a mesh.

    Returns:
        Dictionary with dimensions, bounds, vertex/face count, surface area,
        volume, and watertight / 3D-printability status.
    """
    if isinstance(mesh_or_path, str):
        mesh = load_mesh(mesh_or_path)
    else:
        mesh = mesh_or_path

    # Extract vertex and face counts
    v_count = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
    f_count = len(mesh.faces) if hasattr(mesh, "faces") else 0

    # Bounds & Extents
    if hasattr(mesh, "extents") and mesh.extents is not None and len(mesh.extents) == 3:
        dx, dy, dz = float(mesh.extents[0]), float(mesh.extents[1]), float(mesh.extents[2])
    elif hasattr(mesh, "bounds") and mesh.bounds is not None:
        b = mesh.bounds
        dx, dy, dz = float(b[1][0] - b[0][0]), float(b[1][1] - b[0][1]), float(b[1][2] - b[0][2])
    elif hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
        v = np.asarray(mesh.vertices)
        min_v = v.min(axis=0)
        max_v = v.max(axis=0)
        dx, dy, dz = float(max_v[0] - min_v[0]), float(max_v[1] - min_v[1]), float(max_v[2] - min_v[2])
    else:
        dx, dy, dz = 0.0, 0.0, 0.0

    # Watertightness
    is_watertight = bool(getattr(mesh, "is_watertight", False))

    # Surface Area
    area_mm2 = float(getattr(mesh, "area", 0.0))
    area_cm2 = area_mm2 / 100.0

    # Volume
    try:
        vol_mm3 = float(getattr(mesh, "volume", 0.0))
        vol_cm3 = vol_mm3 / 1000.0
    except Exception:
        vol_mm3 = 0.0
        vol_cm3 = 0.0

    # Euler Characteristic (V - E + F = 2 for closed manifold sphere topology)
    euler_num = getattr(mesh, "euler_number", None)

    return {
        "vertex_count": v_count,
        "triangle_count": f_count,
        "face_count": f_count,
        "dimensions_mm": {
            "x": round(dx, 2),
            "y": round(dy, 2),
            "z": round(dz, 2),
            "formatted": f"{dx:.1f} × {dy:.1f} × {dz:.1f} mm"
        },
        "is_watertight": is_watertight,
        "surface_area_mm2": round(area_mm2, 2),
        "surface_area_cm2": round(area_cm2, 2),
        "volume_mm3": round(vol_mm3, 2),
        "volume_cm3": round(vol_cm3, 2),
        "euler_number": euler_num
    }


def render_3d_mesh_plotly(
    mesh_or_path: Union[str, Any],
    color_theme: str = "Viridis",
    lighting_preset: str = "aerospace",
    flat_shading: bool = True
) -> Any:
    """
    Renders an interactive 3D WebGL viewport using Plotly Mesh3d.

    Args:
        mesh_or_path: File path (str) to .stl / .obj or loaded mesh object.
        color_theme: Plotly continuous colorscale ('Viridis', 'Plasma', 'Cividis', 'Turbo', 'Blues', etc.)
        lighting_preset: 'aerospace', 'metallic', or 'soft'.
        flat_shading: Whether to enable flat polygonal faceted shading.

    Returns:
        plotly.graph_objects.Figure object ready for st.plotly_chart().
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly library is not installed. Cannot render 3D mesh.")

    if isinstance(mesh_or_path, str):
        mesh = load_mesh(mesh_or_path)
    else:
        mesh = mesh_or_path

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        # Return empty placeholder figure
        fig = go.Figure()
        fig.add_annotation(
            text="No mesh geometry available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="gray")
        )
        return fig

    # Separate coordinate components
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # Lighting configurations
    lighting_configs = {
        "aerospace": dict(
            ambient=0.45,
            diffuse=0.75,
            specular=0.40,
            roughness=0.25,
            fresnel=0.20
        ),
        "metallic": dict(
            ambient=0.25,
            diffuse=0.55,
            specular=0.85,
            roughness=0.15,
            fresnel=0.50
        ),
        "soft": dict(
            ambient=0.60,
            diffuse=0.60,
            specular=0.10,
            roughness=0.80,
            fresnel=0.10
        )
    }
    lighting = lighting_configs.get(lighting_preset.lower(), lighting_configs["aerospace"])

    # Color intensity based on Z coordinate for continuous gradient
    intensity = z

    mesh_trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        intensity=intensity,
        colorscale=color_theme,
        flatshading=flat_shading,
        lighting=lighting,
        lightposition=dict(x=150, y=250, z=200),
        colorbar=dict(
            title=dict(text="Z (mm)", side="right"),
            thickness=12,
            len=0.7,
            x=1.02
        ),
        name="PicoGK Model",
        hoverinfo="skip"
    )

    fig = go.Figure(data=[mesh_trace])

    # Dynamic camera positioning based on bounding box
    dx, dy, dz = float(x.max() - x.min()), float(y.max() - y.min()), float(z.max() - z.min())
    max_span = max(dx, dy, dz, 1.0)
    center = [float(x.mean()), float(y.mean()), float(z.mean())]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="X (mm)", font=dict(size=12, color="#888888")),
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zerolinecolor="rgba(128, 128, 128, 0.4)",
                backgroundcolor="rgba(0,0,0,0)"
            ),
            yaxis=dict(
                title=dict(text="Y (mm)", font=dict(size=12, color="#888888")),
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zerolinecolor="rgba(128, 128, 128, 0.4)",
                backgroundcolor="rgba(0,0,0,0)"
            ),
            zaxis=dict(
                title=dict(text="Z (mm)", font=dict(size=12, color="#888888")),
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zerolinecolor="rgba(128, 128, 128, 0.4)",
                backgroundcolor="rgba(0,0,0,0)"
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.4, y=1.4, z=1.2),
                center=dict(x=0, y=0, z=0)
            )
        ),
        margin=dict(l=10, r=10, b=10, t=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550
    )

    return fig
