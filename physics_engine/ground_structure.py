"""
Ground Structure Generative Physics Engine.
Deterministic layout and cross-section optimization for 2D discrete structural grids.
Formulates the plastic limit analysis / Michell truss problem as a Linear Program (LP)
solved deterministically via HiGHS (SciPy).
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linprog


@dataclass
class Node2D:
    """Represents a discrete grid node in 2D space."""
    id: int
    ix: int  # integer grid index along x
    iy: int  # integer grid index along y
    x: float  # physical coordinate in mm
    y: float  # physical coordinate in mm
    is_fixed_x: bool = False
    is_fixed_y: bool = False
    fx: float = 0.0  # External force in N (positive along +x)
    fy: float = 0.0  # External force in N (positive along +y)


@dataclass
class Member2D:
    """Represents a potential candidate truss member connecting two nodes."""
    id: int
    node_i: int
    node_j: int
    length: float  # Length in mm
    cx: float  # Direction cosine along x: (x_j - x_i) / L
    cy: float  # Direction cosine along y: (y_j - y_i) / L
    force: float = 0.0  # Axial force in N (positive = tension, negative = compression)
    area: float = 0.0  # Cross-sectional area in mm^2 (A = |force| / sigma_max)
    radius: float = 0.0  # Equivalent circular strut radius in mm
    stress: float = 0.0  # Axial stress in MPa
    is_active: bool = False  # True if member carries significant load


@dataclass
class GroundStructureResult:
    """Comprehensive result of the Ground Structure deterministic optimization."""
    success: bool
    status_message: str
    nodes: List[Node2D]
    all_members: List[Member2D]
    active_members: List[Member2D]
    active_node_ids: Set[int]
    total_mass_kg: float
    total_volume_mm3: float
    reactions: Dict[int, Tuple[float, float]]  # node_id -> (Rx, Ry) in N
    max_axial_force_N: float
    max_stress_MPa: float
    global_fx_residual: float
    global_fy_residual: float
    global_moment_residual_Nmm: float
    max_nodal_residual_N: float


class GroundStructure2D:
    """
    2D Ground Structure Generator & Determinstic LP Optimizer.
    Synthesizes optimal load paths from an empty discrete domain.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 10.0,
        dy: float = 10.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        """
        Initialize a discrete 2D grid domain.

        Args:
            nx: Number of nodes along x (e.g. 11 for indices 0..10)
            ny: Number of nodes along y (e.g. 5 for indices 0..4)
            dx: Grid spacing along x in mm
            dy: Grid spacing along y in mm
            origin_x: Origin x coordinate
            origin_y: Origin y coordinate
        """
        if nx < 2 or ny < 2:
            raise ValueError(f"Grid dimensions must be at least 2x2. Received nx={nx}, ny={ny}")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.nodes: List[Node2D] = []
        self._node_map: Dict[Tuple[int, int], int] = {}
        self.members: List[Member2D] = []

        self._build_nodes()

    def _build_nodes(self):
        """Create regular node grid with indexing id = ix * ny + iy."""
        node_id = 0
        for ix in range(self.nx):
            for iy in range(self.ny):
                x = self.origin_x + ix * self.dx
                y = self.origin_y + iy * self.dy
                node = Node2D(id=node_id, ix=ix, iy=iy, x=x, y=y)
                self.nodes.append(node)
                self._node_map[(ix, iy)] = node_id
                node_id += 1

    def get_node_id(self, ix: int, iy: int) -> int:
        """Lookup node id from integer grid indices."""
        if (ix, iy) not in self._node_map:
            raise KeyError(f"Grid coordinates ({ix}, {iy}) out of range [0..{self.nx-1}, 0..{self.ny-1}]")
        return self._node_map[(ix, iy)]

    def generate_candidate_members(self, max_span: Optional[int] = None):
        """
        Generate candidate connectivity between grid nodes without redundant collinear overlapping bars.

        A member between (x1, y1) and (x2, y2) is valid iff gcd(|x2-x1|, |y2-y1|) == 1.
        This guarantees no other grid node lies strictly along the segment.
        """
        self.members.clear()
        member_id = 0

        num_nodes = len(self.nodes)
        for i in range(num_nodes):
            n_i = self.nodes[i]
            for j in range(i + 1, num_nodes):
                n_j = self.nodes[j]

                d_ix = abs(n_j.ix - n_i.ix)
                d_iy = abs(n_j.iy - n_i.iy)

                # Skip if span exceeds max_span
                if max_span is not None:
                    if max(d_ix, d_iy) > max_span:
                        continue

                # Collinear overlap check: gcd must be 1
                g = math.gcd(d_ix, d_iy)
                if g != 1:
                    # Intermediate grid node exists on this segment
                    continue

                dx_phys = n_j.x - n_i.x
                dy_phys = n_j.y - n_i.y
                length = math.hypot(dx_phys, dy_phys)
                cx = dx_phys / length
                cy = dy_phys / length

                member = Member2D(
                    id=member_id,
                    node_i=i,
                    node_j=j,
                    length=length,
                    cx=cx,
                    cy=cy,
                )
                self.members.append(member)
                member_id += 1

    def clear_supports(self):
        """Clear all fixed support boundary conditions."""
        for n in self.nodes:
            n.is_fixed_x = False
            n.is_fixed_y = False

    def clear_loads(self):
        """Clear all external point loads."""
        for n in self.nodes:
            n.fx = 0.0
            n.fy = 0.0

    def set_support(self, ix: int, iy: int, fix_x: bool = True, fix_y: bool = True):
        """Fix displacement degrees of freedom for node at (ix, iy)."""
        nid = self.get_node_id(ix, iy)
        self.nodes[nid].is_fixed_x = fix_x
        self.nodes[nid].is_fixed_y = fix_y

    def set_supports_list(self, coords: List[Tuple[int, int]], fix_x: bool = True, fix_y: bool = True):
        """Batch set supports for a list of (ix, iy) coordinates."""
        for (ix, iy) in coords:
            self.set_support(ix, iy, fix_x=fix_x, fix_y=fix_y)

    def add_load(self, ix: int, iy: int, fx: float, fy: float):
        """Apply an external point load in Newtons to node at (ix, iy)."""
        nid = self.get_node_id(ix, iy)
        self.nodes[nid].fx += fx
        self.nodes[nid].fy += fy

    def add_loads_list(self, loads: List[Tuple[int, int, float, float]]):
        """Batch add loads from list of (ix, iy, fx, fy)."""
        for (ix, iy, fx, fy) in loads:
            self.add_load(ix, iy, fx, fy)

    def solve(
        self,
        sigma_t: float = 240.0,
        sigma_c: float = 240.0,
        density_g_cm3: float = 2.70,
        force_threshold_rel: float = 1e-4,
    ) -> GroundStructureResult:
        """
        Solve deterministic Michell ground structure LP problem.

        Args:
            sigma_t: Allowable tensile stress in MPa (N/mm^2)
            sigma_c: Allowable compressive stress in MPa (N/mm^2)
            density_g_cm3: Material density in g/cm^3 (e.g. 2.7 for Aluminium)
            force_threshold_rel: Relative force threshold to consider member active

        Returns:
            GroundStructureResult containing active load paths, sections, and equilibrium metrics.
        """
        if len(self.members) == 0:
            raise RuntimeError("No candidate members generated. Call generate_candidate_members() first.")

        # Density in kg/mm^3: 1 g/cm^3 = 1e-6 kg/mm^3
        density_kg_mm3 = density_g_cm3 * 1e-6

        num_nodes = len(self.nodes)
        num_members = len(self.members)

        # 1. Assemble Equilibrium Matrix B
        # Rows: 2 * num_nodes (DOF 2*i: x, DOF 2*i+1: y)
        # Cols: num_members
        # Member (i, j) pulls node i with +c, pulls node j with -c.
        # Resistance on node: -F_int = F_ext => B * t = F_ext
        # For node i: entry is -cx, -cy. For node j: entry is +cx, +cy.
        B_full = np.zeros((2 * num_nodes, num_members), dtype=np.float64)
        F_full = np.zeros(2 * num_nodes, dtype=np.float64)

        for e_idx, m in enumerate(self.members):
            i = m.node_i
            j = m.node_j
            # Node i:
            B_full[2 * i, e_idx] = -m.cx
            B_full[2 * i + 1, e_idx] = -m.cy
            # Node j:
            B_full[2 * j, e_idx] = +m.cx
            B_full[2 * j + 1, e_idx] = +m.cy

        for n in self.nodes:
            F_full[2 * n.id] = n.fx
            F_full[2 * n.id + 1] = n.fy

        # 2. Partition Free vs Fixed DOFs
        free_dofs: List[int] = []
        fixed_dofs: List[int] = []

        for n in self.nodes:
            if n.is_fixed_x:
                fixed_dofs.append(2 * n.id)
            else:
                free_dofs.append(2 * n.id)

            if n.is_fixed_y:
                fixed_dofs.append(2 * n.id + 1)
            else:
                free_dofs.append(2 * n.id + 1)

        if len(fixed_dofs) < 3:
            raise ValueError(f"System has only {len(fixed_dofs)} fixed DOFs. Rigid body motions possible (need >= 3).")

        B_free = B_full[free_dofs, :]
        F_free = F_full[free_dofs]

        # 3. Formulate Linear Program
        # Variables: x = [t_plus; t_minus] of size 2 * num_members, x >= 0
        # t = t_plus - t_minus
        # Objective: min sum L_e * (t_plus / sigma_t + t_minus / sigma_c)
        lengths = np.array([m.length for m in self.members], dtype=np.float64)
        c_obj = np.concatenate([lengths / sigma_t, lengths / sigma_c])

        # Constraint: [B_free, -B_free] * [t_plus; t_minus] = F_free
        A_eq = np.hstack([B_free, -B_free])
        b_eq = F_free

        # Solve via deterministic HiGHS simplex/interior-point solver
        lp_res = linprog(
            c=c_obj,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )

        if not lp_res.success:
            return GroundStructureResult(
                success=False,
                status_message=f"LP Solver failed: {lp_res.message}",
                nodes=self.nodes,
                all_members=self.members,
                active_members=[],
                active_node_ids=set(),
                total_mass_kg=0.0,
                total_volume_mm3=0.0,
                reactions={},
                max_axial_force_N=0.0,
                max_stress_MPa=0.0,
                global_fx_residual=0.0,
                global_fy_residual=0.0,
                global_moment_residual_Nmm=0.0,
                max_nodal_residual_N=0.0,
            )

        # 4. Extract Solution
        x_sol = lp_res.x
        t_plus = x_sol[:num_members]
        t_minus = x_sol[num_members:]
        t_forces = t_plus - t_minus

        max_force = float(np.max(np.abs(t_forces)))
        abs_threshold = force_threshold_rel * (max_force if max_force > 1e-12 else 1.0)

        active_members: List[Member2D] = []
        active_nodes: Set[int] = set()
        total_vol_mm3 = 0.0

        for idx, m in enumerate(self.members):
            force = float(t_forces[idx])
            m.force = force

            # Member sizing: A = |t| / sigma
            if force >= 0:
                area = force / sigma_t
                stress = force / area if area > 1e-12 else 0.0
            else:
                area = (-force) / sigma_c
                stress = (-force) / area if area > 1e-12 else 0.0

            m.area = float(area)
            m.radius = math.sqrt(area / math.pi) if area > 0 else 0.0
            m.stress = float(stress)

            if abs(force) > abs_threshold and area > 1e-9:
                m.is_active = True
                active_members.append(m)
                active_nodes.add(m.node_i)
                active_nodes.add(m.node_j)
                total_vol_mm3 += area * m.length
            else:
                m.is_active = False

        total_mass_kg = total_vol_mm3 * density_kg_mm3

        # 5. Compute Reactions & Check Global Equilibrium
        # R = - B_fixed * t (such that R + B_fixed * t = 0)
        # Nodal balance everywhere: B_full * t - F_full - R_full = 0
        B_times_t = B_full @ t_forces
        reactions: Dict[int, Tuple[float, float]] = {}

        for n in self.nodes:
            rx = 0.0
            ry = 0.0
            if n.is_fixed_x:
                rx = float(B_times_t[2 * n.id] - n.fx)
            if n.is_fixed_y:
                ry = float(B_times_t[2 * n.id + 1] - n.fy)

            if n.is_fixed_x or n.is_fixed_y:
                reactions[n.id] = (rx, ry)

        # Equilibrium residuals
        nodal_residual = np.abs(B_free @ t_forces - F_free)
        max_nodal_res = float(np.max(nodal_residual)) if len(nodal_residual) > 0 else 0.0

        # Global force and moment sum
        sum_fx = sum(n.fx for n in self.nodes) + sum(r[0] for r in reactions.values())
        sum_fy = sum(n.fy for n in self.nodes) + sum(r[1] for r in reactions.values())

        # Moment about origin (0, 0)
        sum_m = 0.0
        for n in self.nodes:
            # r x F = x * Fy - y * Fx
            sum_m += (n.x * n.fy - n.y * n.fx)
        for nid, (rx, ry) in reactions.items():
            node = self.nodes[nid]
            sum_m += (node.x * ry - node.y * rx)

        max_stress = float(max((m.stress for m in active_members), default=0.0))

        return GroundStructureResult(
            success=True,
            status_message="Optimal layout converged successfully via HiGHS LP.",
            nodes=self.nodes,
            all_members=self.members,
            active_members=active_members,
            active_node_ids=active_nodes,
            total_mass_kg=float(total_mass_kg),
            total_volume_mm3=float(total_vol_mm3),
            reactions=reactions,
            max_axial_force_N=max_force,
            max_stress_MPa=max_stress,
            global_fx_residual=float(abs(sum_fx)),
            global_fy_residual=float(abs(sum_fy)),
            global_moment_residual_Nmm=float(abs(sum_m)),
            max_nodal_residual_N=max_nodal_res,
        )

    def to_plotly_figure(
        self,
        res: GroundStructureResult,
        show_all_nodes: bool = True,
        scale_thickness: bool = True
    ):
        """
        Generates an interactive Plotly 2D figure with tension/compression coloring,
        proportional beam thickness, support markers, and load vectors.
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # 1. Draw Active Members
        for m in res.active_members:
            n1 = self.nodes[m.node_i]
            n2 = self.nodes[m.node_j]
            is_tension = m.force > 0
            color = "#38bdf8" if is_tension else "#f43f5e"
            width = max(1.2, min(7.0, m.area * 0.35)) if scale_thickness else 2.0
            kind = "TRAZIONE" if is_tension else "COMPRESSIONE"

            hover_text = (
                f"<b>Asta #{m.id}</b> ({kind})<br>"
                f"Forza: {m.force:+.2f} N<br>"
                f"Sezione: {m.area:.3f} mm²<br>"
                f"Raggio: {m.radius:.3f} mm<br>"
                f"Tensione: {m.stress:.1f} MPa"
            )

            fig.add_trace(go.Scatter(
                x=[n1.x, n2.x],
                y=[n1.y, n2.y],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False
            ))

        # 2. Draw Nodes
        fixed_x, fixed_y, fixed_txt = [], [], []
        loaded_x, loaded_y, loaded_txt = [], [], []
        active_x, active_y, active_txt = [], [], []
        inactive_x, inactive_y = [], []

        for n in self.nodes:
            is_active = n.id in res.active_node_ids
            has_load = (abs(n.fx) > 1e-6 or abs(n.fy) > 1e-6)

            if n.is_fixed_x or n.is_fixed_y:
                rx, ry = res.reactions.get(n.id, (0.0, 0.0))
                fixed_x.append(n.x)
                fixed_y.append(n.y)
                fixed_txt.append(f"<b>Incastro ({n.ix}, {n.iy})</b><br>Rx: {rx:+.1f} N<br>Ry: {ry:+.1f} N")
            elif has_load:
                loaded_x.append(n.x)
                loaded_y.append(n.y)
                loaded_txt.append(f"<b>Carico ({n.ix}, {n.iy})</b><br>Fx: {n.fx} N<br>Fy: {n.fy} N")
            elif is_active:
                active_x.append(n.x)
                active_y.append(n.y)
                active_txt.append(f"Nodo Attivo ({n.ix}, {n.iy})")
            elif show_all_nodes:
                inactive_x.append(n.x)
                inactive_y.append(n.y)

        # Inactive grid dots
        if inactive_x:
            fig.add_trace(go.Scatter(
                x=inactive_x, y=inactive_y,
                mode="markers",
                marker=dict(size=4, color="#475569"),
                hoverinfo="none",
                name="Vuoto (Inattivo)"
            ))

        # Active nodes
        if active_x:
            fig.add_trace(go.Scatter(
                x=active_x, y=active_y,
                mode="markers",
                marker=dict(size=6, color="#93c5fd", line=dict(color="#1e3a8a", width=1)),
                hoverinfo="text",
                hovertext=active_txt,
                name="Nodi Attivi"
            ))

        # Fixed supports
        if fixed_x:
            fig.add_trace(go.Scatter(
                x=fixed_x, y=fixed_y,
                mode="markers",
                marker=dict(size=12, symbol="square", color="#f59e0b", line=dict(color="#78350f", width=1.5)),
                hoverinfo="text",
                hovertext=fixed_txt,
                name="Incastro (u=v=0)"
            ))

        # Loaded nodes
        if loaded_x:
            fig.add_trace(go.Scatter(
                x=loaded_x, y=loaded_y,
                mode="markers",
                marker=dict(size=12, symbol="circle", color="#ef4444", line=dict(color="#ffffff", width=1.5)),
                hoverinfo="text",
                hovertext=loaded_txt,
                name="Carico Applicato"
            ))

        # Add load arrows (annotations)
        for n in self.nodes:
            if abs(n.fx) > 1e-6 or abs(n.fy) > 1e-6:
                mag = math.hypot(n.fx, n.fy)
                ax = 0.0 if abs(n.fx) < 1e-6 else (-30.0 * n.fx / mag)
                ay = 0.0 if abs(n.fy) < 1e-6 else (30.0 * n.fy / mag)
                fig.add_annotation(
                    x=n.x, y=n.y,
                    ax=ax, ay=ay,
                    xref="x", yref="y",
                    axref="pixel", ayref="pixel",
                    text=f"F=({n.fx:.0f}, {n.fy:.0f})N",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="#ef4444",
                    font=dict(color="#f87171", size=10, family="monospace")
                )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#020617",
            xaxis=dict(
                title="X (mm)",
                scaleanchor="y",
                scaleratio=1,
                gridcolor="#1e293b",
                zerolinecolor="#334155"
            ),
            yaxis=dict(
                title="Y (mm)",
                gridcolor="#1e293b",
                zerolinecolor="#334155"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=420
        )

        return fig

    def to_truss_params(
        self,
        res: GroundStructureResult,
        voxel_size_mm: float = 0.5,
        depth_z_mm: float = 10.0
    ):
        """
        Converts active members to GroundStructureTrussParams for PicoGK 3D voxelization.
        """
        from PicoGK_Dashboard.core.models import BeamParam, GroundStructureTrussParams

        beams = []
        for m in res.active_members:
            n1 = self.nodes[m.node_i]
            n2 = self.nodes[m.node_j]
            beams.append(BeamParam(
                x1=float(n1.x),
                y1=float(n1.y),
                z1=0.0,
                x2=float(n2.x),
                y2=float(n2.y),
                z2=0.0,
                radius=float(m.radius),
                force=float(m.force)
            ))

        return GroundStructureTrussParams(
            voxel_size_mm=voxel_size_mm,
            depth_z_mm=depth_z_mm,
            beams=beams
        )
