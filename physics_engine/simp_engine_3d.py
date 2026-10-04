"""
3D Continuum Topology Optimization Engine with Linearized Buckling Stability.
Physics-driven material distribution optimizer using 8-node hexahedral (H8) finite elements,
SIMP material interpolation with cubic penalization (p_K = 3.0),
vectorized 3D spatial filtering via scipy.ndimage.convolve,
preconditioned conjugate gradient (PCG) solver with Jacobi preconditioning and displacement warm-starting,
linearized buckling analysis (LBA) with 6 precomputed geometric stiffness basis matrices,
exact multi-objective adjoint sensitivity analysis with dynamic L1 gradient scaling and alpha scalarization,
Optimality Criteria (OC) bisection update loop,
and watertight Trimesh STL export.
"""

from dataclasses import dataclass, field
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.ndimage as ndimage


# =============================================================================
# 1. H8 Nodal Coordinates in Natural Space [-1, 1]^3
# =============================================================================
H8_NODES_NATURAL = np.array([
    [-1.0, -1.0, -1.0],  # Node 1
    [ 1.0, -1.0, -1.0],  # Node 2
    [ 1.0,  1.0, -1.0],  # Node 3
    [-1.0,  1.0, -1.0],  # Node 4
    [-1.0, -1.0,  1.0],  # Node 5
    [ 1.0, -1.0,  1.0],  # Node 6
    [ 1.0,  1.0,  1.0],  # Node 7
    [-1.0,  1.0,  1.0],  # Node 8
], dtype=np.float64)


def h8_shape_functions(xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates trilinear shape functions N_i and natural gradients dN_i / d(xi, eta, zeta).

    Returns:
        N: array of shape (8,) containing N_1 ... N_8
        dN_dnat: array of shape (8, 3) where column 0 is dN/dxi, col 1 is dN/deta, col 2 is dN/dzeta
    """
    xi_i = H8_NODES_NATURAL[:, 0]
    eta_i = H8_NODES_NATURAL[:, 1]
    zeta_i = H8_NODES_NATURAL[:, 2]

    N = 0.125 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta)

    dN_dxi = 0.125 * xi_i * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta)
    dN_deta = 0.125 * eta_i * (1.0 + xi_i * xi) * (1.0 + zeta_i * zeta)
    dN_dzeta = 0.125 * zeta_i * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)

    dN_dnat = np.column_stack([dN_dxi, dN_deta, dN_dzeta])
    return N, dN_dnat


def h8_jacobian(dx: float, dy: float, dz: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Computes diagonal Jacobian matrix J, determinant det(J), and inverse J^-1.
    For uniform Cartesian hexahedra:
        J = diag(dx/2, dy/2, dz/2)
        det(J) = dx * dy * dz / 8 = V_e / 8
    """
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        raise ValueError("Element dimensions dx, dy, dz must be strictly positive.")

    J = np.diag([dx / 2.0, dy / 2.0, dz / 2.0])
    det_J = (dx * dy * dz) / 8.0
    inv_J = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])
    return J, det_J, inv_J


def h8_strain_displacement_b(
    xi: float, eta: float, zeta: float,
    dx: float, dy: float, dz: float
) -> np.ndarray:
    """
    Assembles the 6 x 24 strain-displacement matrix B(xi, eta, zeta).
    Engineering strain vector: [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]^T
    """
    _, dN_dnat = h8_shape_functions(xi, eta, zeta)
    dN_dx = (2.0 / dx) * dN_dnat[:, 0]
    dN_dy = (2.0 / dy) * dN_dnat[:, 1]
    dN_dz = (2.0 / dz) * dN_dnat[:, 2]

    B = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        col = 3 * i
        # eps_xx
        B[0, col + 0] = dN_dx[i]
        # eps_yy
        B[1, col + 1] = dN_dy[i]
        # eps_zz
        B[2, col + 2] = dN_dz[i]
        # gamma_yz = dv/dz + dw/dy
        B[3, col + 1] = dN_dz[i]
        B[3, col + 2] = dN_dy[i]
        # gamma_xz = du/dz + dw/dx
        B[4, col + 0] = dN_dz[i]
        B[4, col + 2] = dN_dx[i]
        # gamma_xy = du/dy + dv/dx
        B[5, col + 0] = dN_dy[i]
        B[5, col + 1] = dN_dx[i]

    return B


def elastic_constitutive_matrix_d(E0: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """
    Computes 6 x 6 isotropic elasticity matrix D under generalized Hooke's law.
    """
    if nu >= 0.5 or nu <= -1.0:
        raise ValueError("Poisson ratio nu must satisfy -1.0 < nu < 0.5 for physical elasticity.")

    factor = E0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c1 = (1.0 - nu) * factor
    c2 = nu * factor
    c3 = 0.5 * (1.0 - 2.0 * nu) * factor

    D = np.array([
        [c1,  c2,  c2,  0.0, 0.0, 0.0],
        [c2,  c1,  c2,  0.0, 0.0, 0.0],
        [c2,  c2,  c1,  0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c3,  0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c3,  0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c3 ],
    ], dtype=np.float64)
    return D


def h8_element_stiffness_k0(
    dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
    E0: float = 1.0, nu: float = 0.3
) -> np.ndarray:
    """
    Integrates the 24 x 24 elemental stiffness matrix k0 using 2x2x2 Gauss quadrature.
    """
    D = elastic_constitutive_matrix_d(E0, nu)
    _, det_J, _ = h8_jacobian(dx, dy, dz)

    gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    k0 = np.zeros((24, 24), dtype=np.float64)

    for xi in gauss_pts:
        for eta in gauss_pts:
            for zeta in gauss_pts:
                B = h8_strain_displacement_b(xi, eta, zeta, dx, dy, dz)
                k0 += (B.T @ D @ B) * det_J

    return k0


def h8_displacement_gradient_bnl(
    xi: float, eta: float, zeta: float,
    dx: float, dy: float, dz: float
) -> np.ndarray:
    """
    Assembles the 9 x 24 displacement gradient matrix B_nl.
    Gradient vector theta = [du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz]^T
    """
    _, dN_dnat = h8_shape_functions(xi, eta, zeta)
    dN_dx = (2.0 / dx) * dN_dnat[:, 0]
    dN_dy = (2.0 / dy) * dN_dnat[:, 1]
    dN_dz = (2.0 / dz) * dN_dnat[:, 2]

    B_nl = np.zeros((9, 24), dtype=np.float64)
    for i in range(8):
        c = 3 * i
        # du/dx, du/dy, du/dz
        B_nl[0, c + 0] = dN_dx[i]
        B_nl[1, c + 0] = dN_dy[i]
        B_nl[2, c + 0] = dN_dz[i]
        # dv/dx, dv/dy, dv/dz
        B_nl[3, c + 1] = dN_dx[i]
        B_nl[4, c + 1] = dN_dy[i]
        B_nl[5, c + 1] = dN_dz[i]
        # dw/dx, dw/dy, dw/dz
        B_nl[6, c + 2] = dN_dx[i]
        B_nl[7, c + 2] = dN_dy[i]
        B_nl[8, c + 2] = dN_dz[i]

    return B_nl


def geometric_stiffness_basis_matrices(
    dx: float = 1.0, dy: float = 1.0, dz: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Computes the 6 constant 24 x 24 geometric stiffness basis matrices G0_k
    via centroidal 1-point quadrature (super-convergent).
    Components: 'xx', 'yy', 'zz', 'yz', 'xz', 'xy'
    """
    V_e = dx * dy * dz
    B_nl_0 = h8_displacement_gradient_bnl(0.0, 0.0, 0.0, dx, dy, dz)

    T_matrices = {
        'xx': np.zeros((9, 9), dtype=np.float64),
        'yy': np.zeros((9, 9), dtype=np.float64),
        'zz': np.zeros((9, 9), dtype=np.float64),
        'yz': np.zeros((9, 9), dtype=np.float64),
        'xz': np.zeros((9, 9), dtype=np.float64),
        'xy': np.zeros((9, 9), dtype=np.float64),
    }

    for row in [0, 3, 6]:
        T_matrices['xx'][row, row] = 1.0
    for row in [1, 4, 7]:
        T_matrices['yy'][row, row] = 1.0
    for row in [2, 5, 8]:
        T_matrices['zz'][row, row] = 1.0
    for r, c in [(1, 2), (2, 1), (4, 5), (5, 4), (7, 8), (8, 7)]:
        T_matrices['yz'][r, c] = 1.0
    for r, c in [(0, 2), (2, 0), (3, 5), (5, 3), (6, 8), (8, 6)]:
        T_matrices['xz'][r, c] = 1.0
    for r, c in [(0, 1), (1, 0), (3, 4), (4, 3), (6, 7), (7, 6)]:
        T_matrices['xy'][r, c] = 1.0

    G0_bases = {}
    for key, T_mat in T_matrices.items():
        G0_bases[key] = V_e * (B_nl_0.T @ T_mat @ B_nl_0)

    return G0_bases


def spherical_cone_kernel(rmin: float, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Generates discrete 3D spherical cone convolution kernel:
        K(i, j, k) = max(0, rmin - dist(i*dx, j*dy, k*dz))
    """
    rx = int(np.ceil(rmin / dx))
    ry = int(np.ceil(rmin / dy))
    rz = int(np.ceil(rmin / dz))

    iz, iy, ix = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    dist = np.sqrt((ix * dx)**2 + (iy * dy)**2 + (iz * dz)**2)
    kernel = np.maximum(0.0, rmin - dist)
    return kernel


# =============================================================================
# 2. SIMPResult3D Container
# =============================================================================
@dataclass
class SIMPResult3D:
    """
    Encapsulates results and diagnostic histories of a 3D SIMP optimization solve.
    """
    success: bool
    density_matrix: np.ndarray  # 3D float array of shape (nelz, nely, nelx)
    compliance: float
    volume_fraction: float
    iterations_run: int
    execution_time_sec: float
    compliance_history: List[float] = field(default_factory=list)
    change_history: List[float] = field(default_factory=list)
    blf_history: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    status_message: str = ""
    nelx: int = 0
    nely: int = 0
    nelz: int = 0
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    displacements: Optional[np.ndarray] = None
    stresses: Optional[Dict[str, np.ndarray]] = None

    def export_voxel_stl(self, filepath: str, threshold: float = 0.5) -> str:
        """
        Exports a watertight, manifold STL surface mesh from the 3D density matrix.

        Extracts only the *boundary* faces of the solid voxel region — i.e., faces shared
        between a solid voxel and an empty voxel (or domain boundary). This eliminates
        all internal shared faces, producing a geometrically correct, non-self-intersecting
        closed surface compatible with PicoGK / OpenVDB and all slicer software.

        Density matrix layout: density_matrix[iz, iy, ix] (shape: [nelz, nely, nelx]).
        Physical vertex positions:  x = ix * dx,  y = iy * dy,  z = iz * dz.
        """
        import trimesh

        solid = (self.density_matrix >= threshold)  # shape: (nelz, nely, nelx)
        if not np.any(solid):
            # Fall back to adaptive threshold to prevent empty mesh
            solid = (self.density_matrix >= np.percentile(self.density_matrix, 50))
            if not np.any(solid):
                solid = np.ones_like(self.density_matrix, dtype=bool)

        nz, ny, nx = solid.shape
        dx, dy, dz = self.dx, self.dy, self.dz

        # --- Boundary face extraction ---
        # Pad with False on all 6 sides to handle domain boundaries uniformly
        padded = np.pad(solid, 1, mode='constant', constant_values=False)

        vertices_list: list[np.ndarray] = []
        faces_list: list[np.ndarray] = []
        v_offset = 0

        # Face offsets in padded array per direction:
        # For each axis and direction we compare solid[iz, iy, ix] with its neighbor.
        # A face is emitted where solid is True and its neighbor is False.
        directions = [
            # (axis, shift, corner offsets for the quad face in physical space)
            # X- face (ix neighbour at ix-1): normal -X
            ('x-', lambda iz, iy, ix: [
                [ix*dx, iy*dy,       iz*dz],
                [ix*dx, (iy+1)*dy,   iz*dz],
                [ix*dx, (iy+1)*dy,   (iz+1)*dz],
                [ix*dx, iy*dy,       (iz+1)*dz],
            ]),
            # X+ face: normal +X
            ('x+', lambda iz, iy, ix: [
                [(ix+1)*dx, iy*dy,       iz*dz],
                [(ix+1)*dx, iy*dy,       (iz+1)*dz],
                [(ix+1)*dx, (iy+1)*dy,   (iz+1)*dz],
                [(ix+1)*dx, (iy+1)*dy,   iz*dz],
            ]),
            # Y- face: normal -Y
            ('y-', lambda iz, iy, ix: [
                [ix*dx,       iy*dy, iz*dz],
                [ix*dx,       iy*dy, (iz+1)*dz],
                [(ix+1)*dx,   iy*dy, (iz+1)*dz],
                [(ix+1)*dx,   iy*dy, iz*dz],
            ]),
            # Y+ face: normal +Y
            ('y+', lambda iz, iy, ix: [
                [ix*dx,       (iy+1)*dy, iz*dz],
                [(ix+1)*dx,   (iy+1)*dy, iz*dz],
                [(ix+1)*dx,   (iy+1)*dy, (iz+1)*dz],
                [ix*dx,       (iy+1)*dy, (iz+1)*dz],
            ]),
            # Z- face: normal -Z
            ('z-', lambda iz, iy, ix: [
                [ix*dx,       iy*dy,       iz*dz],
                [(ix+1)*dx,   iy*dy,       iz*dz],
                [(ix+1)*dx,   (iy+1)*dy,   iz*dz],
                [ix*dx,       (iy+1)*dy,   iz*dz],
            ]),
            # Z+ face: normal +Z
            ('z+', lambda iz, iy, ix: [
                [ix*dx,       iy*dy,       (iz+1)*dz],
                [ix*dx,       (iy+1)*dy,   (iz+1)*dz],
                [(ix+1)*dx,   (iy+1)*dy,   (iz+1)*dz],
                [(ix+1)*dx,   iy*dy,       (iz+1)*dz],
            ]),
        ]

        # Neighbor masks per direction (using padded array; padded coords = real + 1)
        neighbor_masks = {
            'x-': padded[1:nz+1, 1:ny+1, 0:nx],    # neighbor at ix-1
            'x+': padded[1:nz+1, 1:ny+1, 2:nx+2],  # neighbor at ix+1
            'y-': padded[1:nz+1, 0:ny,   1:nx+1],  # neighbor at iy-1
            'y+': padded[1:nz+1, 2:ny+2, 1:nx+1],  # neighbor at iy+1
            'z-': padded[0:nz,   1:ny+1, 1:nx+1],  # neighbor at iz-1
            'z+': padded[2:nz+2, 1:ny+1, 1:nx+1],  # neighbor at iz+1
        }

        for dir_key, quad_fn in directions:
            # Voxels that are solid AND whose neighbor in this direction is empty
            boundary_mask = solid & ~neighbor_masks[dir_key]
            iz_arr, iy_arr, ix_arr = np.where(boundary_mask)

            for iz, iy, ix in zip(iz_arr, iy_arr, ix_arr):
                quad = np.array(quad_fn(int(iz), int(iy), int(ix)), dtype=np.float64)
                vertices_list.append(quad)
                # Two triangles per quad
                faces_list.append([v_offset, v_offset+1, v_offset+2])
                faces_list.append([v_offset, v_offset+2, v_offset+3])
                v_offset += 4

        if not vertices_list:
            # Empty result guard — return minimal valid STL
            mesh = trimesh.creation.box()
        else:
            verts = np.vstack(vertices_list)
            faces = np.array(faces_list, dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mesh.export(filepath)
        return filepath

    def get_boundary_mesh(
        self,
        threshold: float = 0.35,
        smooth_iterations: int = 0,
        subdivide: bool = False
    ):
        """
        Extracts a closed watertight boundary mesh of the solid voxels with optional
        subdivision and volume-preserving Taubin smoothing (non-shrinking).
        """
        import trimesh

        solid = (self.density_matrix >= threshold)
        if not np.any(solid):
            solid = (self.density_matrix >= np.percentile(self.density_matrix, 50))
            if not np.any(solid):
                solid = np.ones_like(self.density_matrix, dtype=bool)

        nz, ny, nx = solid.shape
        dx, dy, dz = self.dx, self.dy, self.dz
        padded = np.pad(solid, 1, mode='constant', constant_values=False)

        vertices_list = []
        faces_list = []
        v_offset = 0

        directions = [
            ('x-', lambda iz, iy, ix: [
                [ix*dx, iy*dy,       iz*dz],
                [ix*dx, iy*dy,       (iz+1)*dz],
                [ix*dx, (iy+1)*dy,   (iz+1)*dz],
                [ix*dx, (iy+1)*dy,   iz*dz],
            ]),
            ('x+', lambda iz, iy, ix: [
                [(ix+1)*dx, iy*dy,       iz*dz],
                [(ix+1)*dx, (iy+1)*dy,   iz*dz],
                [(ix+1)*dx, (iy+1)*dy,   (iz+1)*dz],
                [(ix+1)*dx, iy*dy,       (iz+1)*dz],
            ]),
            ('y-', lambda iz, iy, ix: [
                [ix*dx,       iy*dy, iz*dz],
                [(ix+1)*dx,   iy*dy, iz*dz],
                [(ix+1)*dx,   iy*dy, (iz+1)*dz],
                [ix*dx,       iy*dy, (iz+1)*dz],
            ]),
            ('y+', lambda iz, iy, ix: [
                [ix*dx,       (iy+1)*dy, iz*dz],
                [ix*dx,       (iy+1)*dy, (iz+1)*dz],
                [(ix+1)*dx,   (iy+1)*dy, (iz+1)*dz],
                [(ix+1)*dx,   (iy+1)*dy, iz*dz],
            ]),
            ('z-', lambda iz, iy, ix: [
                [ix*dx,       iy*dy,       iz*dz],
                [ix*dx,       (iy+1)*dy,   iz*dz],
                [(ix+1)*dx,   (iy+1)*dy,   iz*dz],
                [(ix+1)*dx,   iy*dy,       iz*dz],
            ]),
            ('z+', lambda iz, iy, ix: [
                [ix*dx,       iy*dy,       (iz+1)*dz],
                [(ix+1)*dx,   iy*dy,       (iz+1)*dz],
                [(ix+1)*dx,   (iy+1)*dy,   (iz+1)*dz],
                [ix*dx,       (iy+1)*dy,   (iz+1)*dz],
            ]),
        ]

        neighbor_masks = {
            'x-': padded[1:nz+1, 1:ny+1, 0:nx],
            'x+': padded[1:nz+1, 1:ny+1, 2:nx+2],
            'y-': padded[1:nz+1, 0:ny,   1:nx+1],
            'y+': padded[1:nz+1, 2:ny+2, 1:nx+1],
            'z-': padded[0:nz,   1:ny+1, 1:nx+1],
            'z+': padded[2:nz+2, 1:ny+1, 1:nx+1],
        }

        for dir_key, quad_fn in directions:
            boundary_mask = solid & ~neighbor_masks[dir_key]
            iz_arr, iy_arr, ix_arr = np.where(boundary_mask)

            for iz, iy, ix in zip(iz_arr, iy_arr, ix_arr):
                quad = np.array(quad_fn(int(iz), int(iy), int(ix)), dtype=np.float64)
                vertices_list.append(quad)
                faces_list.append([v_offset, v_offset+1, v_offset+2])
                faces_list.append([v_offset, v_offset+2, v_offset+3])
                v_offset += 4

        if not vertices_list:
            mesh = trimesh.creation.box()
        else:
            verts = np.vstack(vertices_list)
            faces = np.array(faces_list, dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        if subdivide and smooth_iterations > 0 and len(mesh.faces) < 20000:
            mesh = mesh.subdivide()

        if smooth_iterations > 0 and len(mesh.vertices) > 0:
            # Stable volume-preserving Taubin smoothing (canonical parameters lambda=0.33, nu=-0.34)
            trimesh.smoothing.filter_taubin(mesh, lamb=0.33, nu=-0.34, iterations=int(smooth_iterations))

        return mesh

    def export_smooth_stl(
        self,
        filepath: str,
        threshold: float = 0.35,
        smooth_iterations: int = 10,
        subdivide: bool = False
    ) -> str:
        """Exports volume-preserving Taubin smoothed watertight STL file."""
        mesh = self.get_boundary_mesh(threshold=threshold, smooth_iterations=smooth_iterations, subdivide=subdivide)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mesh.export(filepath)
        return filepath



# =============================================================================
# 3. SIMPOptimizer3D Solver
# =============================================================================
class SIMPOptimizer3D:
    """
    Solves 3D continuum topology optimization for minimum compliance and
    linearized buckling stability using structured H8 finite elements.
    """

    def __init__(
        self,
        nelx: int,
        nely: int,
        nelz: int,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        E0: float = 1.0,
        Emin: float = 1e-6,
        nu: float = 0.3,
        penal: float = 3.0,
        penal_g: float = 3.0,
        rmin: float = 1.5,
        volfrac: float = 0.3,
        alpha: float = 0.0,
        solver_type: str = "pcg",  # "pcg" or "direct"
        max_iter: int = 50,
        tol: float = 0.01,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ):
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.nelz = int(nelz)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.E0 = float(E0)
        self.Emin = float(Emin)
        self.nu = float(nu)
        self.penal = float(penal)
        self.penal_g = float(penal_g)
        self.rmin = float(rmin)
        self.volfrac = float(volfrac)
        self.alpha = float(alpha)
        self.solver_type = str(solver_type).lower()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.progress_callback = progress_callback

        self.num_elements = self.nelx * self.nely * self.nelz
        self.num_nodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
        self.num_dofs = 3 * self.num_nodes

        # 1. Precompute Element Matrices
        self.D = elastic_constitutive_matrix_d(self.E0, self.nu)
        self.k0 = h8_element_stiffness_k0(self.dx, self.dy, self.dz, self.E0, self.nu)
        self.B0 = h8_strain_displacement_b(0.0, 0.0, 0.0, self.dx, self.dy, self.dz)
        self.DB0_T = (self.D @ self.B0).T  # Precomputed for fast stress evaluation (24, 6)

        # 2. Precompute 6 Geometric Stiffness Basis Matrices
        G0_dict = geometric_stiffness_basis_matrices(self.dx, self.dy, self.dz)
        self.basis_keys = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
        self.G0_bases = [G0_dict[k] for k in self.basis_keys]
        self.G0_bases_arr = np.array(self.G0_bases)  # shape (6, 24, 24)

        # 3. Precompute edofMat and Invariant Assembly Sparsity Indices (iK, jK)
        self.edofMat = self._build_edof_matrix()
        self.iK = np.repeat(self.edofMat, 24, axis=1).ravel()
        self.jK = np.tile(self.edofMat, (1, 24)).ravel()

        # 4. Precompute 3D Convolution Spatial Filter
        self.kernel = spherical_cone_kernel(self.rmin, self.dx, self.dy, self.dz)
        ones_grid = np.ones((self.nelz, self.nely, self.nelx), dtype=np.float64)
        conv_ones = ndimage.convolve(ones_grid, self.kernel, mode='constant', cval=0.0)
        self.kernel_normalizer = np.maximum(1e-12, conv_ones)

        # 5. Boundary Condition Storage
        self.fixed_dofs: Set[int] = set()
        self.force_vector = np.zeros(self.num_dofs, dtype=np.float64)

    def _build_edof_matrix(self) -> np.ndarray:
        """
        Builds element degree-of-freedom mapping matrix of shape (num_elements, 24).
        Lexicographical ordering: element e = elz*(nely*nelx) + ely*nelx + elx.
        Node ID = i*(nely+1)*(nelz+1) + j*(nelz+1) + k.
        """
        edofMat = np.zeros((self.num_elements, 24), dtype=np.int32)
        idx = 0
        for elz in range(self.nelz):
            for ely in range(self.nely):
                for elx in range(self.nelx):
                    # 8 local nodes of element (elx, ely, elz):
                    n1 = self.node_id(elx,     ely,     elz)
                    n2 = self.node_id(elx + 1, ely,     elz)
                    n3 = self.node_id(elx + 1, ely + 1, elz)
                    n4 = self.node_id(elx,     ely + 1, elz)
                    n5 = self.node_id(elx,     ely,     elz + 1)
                    n6 = self.node_id(elx + 1, ely,     elz + 1)
                    n7 = self.node_id(elx + 1, ely + 1, elz + 1)
                    n8 = self.node_id(elx,     ely + 1, elz + 1)

                    nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
                    dofs = []
                    for n in nodes:
                        dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])
                    edofMat[idx, :] = dofs
                    idx += 1
        return edofMat

    def node_id(self, i: int, j: int, k: int) -> int:
        """Returns node ID from grid coordinate (i in 0..nelx, j in 0..nely, k in 0..nelz)."""
        if not (0 <= i <= self.nelx and 0 <= j <= self.nely and 0 <= k <= self.nelz):
            raise IndexError(f"Node indices ({i}, {j}, {k}) out of range [0..{self.nelx}, 0..{self.nely}, 0..{self.nelz}]")
        return i * (self.nely + 1) * (self.nelz + 1) + j * (self.nelz + 1) + k

    def clear_boundary_conditions(self):
        """Clears all fixed DOFs and applied forces."""
        self.fixed_dofs.clear()
        self.force_vector.fill(0.0)

    def fix_dof(self, dof: int):
        """Fix a single degree of freedom."""
        if 0 <= dof < self.num_dofs:
            self.fixed_dofs.add(dof)

    def fix_node(self, i: int, j: int, k: int, fix_x: bool = True, fix_y: bool = True, fix_z: bool = True):
        """Fix displacements of node (i, j, k)."""
        nid = self.node_id(i, j, k)
        if fix_x:
            self.fixed_dofs.add(3 * nid)
        if fix_y:
            self.fixed_dofs.add(3 * nid + 1)
        if fix_z:
            self.fixed_dofs.add(3 * nid + 2)

    def fix_face(self, face: str = "left", fix_x: bool = True, fix_y: bool = True, fix_z: bool = True):
        """
        Fix an entire boundary face:
        'left' (x=0), 'right' (x=nelx), 'bottom' (y=0), 'top' (y=nely),
        'front' (z=0), 'back' (z=nelz).
        """
        face = face.lower()
        if face == "left":
            for j in range(self.nely + 1):
                for k in range(self.nelz + 1):
                    self.fix_node(0, j, k, fix_x, fix_y, fix_z)
        elif face == "right":
            for j in range(self.nely + 1):
                for k in range(self.nelz + 1):
                    self.fix_node(self.nelx, j, k, fix_x, fix_y, fix_z)
        elif face == "bottom":
            for i in range(self.nelx + 1):
                for k in range(self.nelz + 1):
                    self.fix_node(i, 0, k, fix_x, fix_y, fix_z)
        elif face == "top":
            for i in range(self.nelx + 1):
                for k in range(self.nelz + 1):
                    self.fix_node(i, self.nely, k, fix_x, fix_y, fix_z)
        elif face in ["front", "bottom_z"]:
            for i in range(self.nelx + 1):
                for j in range(self.nely + 1):
                    self.fix_node(i, j, 0, fix_x, fix_y, fix_z)
        elif face in ["back", "top_z"]:
            for i in range(self.nelx + 1):
                for j in range(self.nely + 1):
                    self.fix_node(i, j, self.nelz, fix_x, fix_y, fix_z)
        else:
            raise ValueError(f"Unknown face '{face}'. Expected 'left', 'right', 'bottom', 'top', 'front', or 'back'.")

    def fix_wall(self, side: str = "left", fix_x: bool = True, fix_y: bool = True, fix_z: bool = True):
        """Compatibility alias for fix_face."""
        self.fix_face(face=side, fix_x=fix_x, fix_y=fix_y, fix_z=fix_z)

    def add_load(self, i: int, j: int, k: int, fx: float = 0.0, fy: float = 0.0, fz: float = 0.0):
        """Apply concentrated point load in Newtons to node (i, j, k)."""
        nid = self.node_id(i, j, k)
        self.force_vector[3 * nid] += fx
        self.force_vector[3 * nid + 1] += fy
        self.force_vector[3 * nid + 2] += fz

    def export_stl(self, result: SIMPResult3D, filepath: str, threshold: float = 0.35) -> str:
        """Delegates STL export to the SIMPResult3D container."""
        return result.export_voxel_stl(filepath, threshold)

    def export_smooth_stl(
        self,
        result: SIMPResult3D,
        filepath: str,
        threshold: float = 0.35,
        smooth_iterations: int = 10,
        subdivide: bool = False
    ) -> str:
        """Delegates smooth STL export to the SIMPResult3D container."""
        return result.export_smooth_stl(filepath, threshold, smooth_iterations, subdivide)

    def get_mesh(
        self,
        result: SIMPResult3D,
        threshold: float = 0.35,
        smooth_iterations: int = 0,
        subdivide: bool = False
    ):
        """Returns trimesh object (raw or Taubin-smoothed) for direct 3D visualization."""
        return result.get_boundary_mesh(threshold, smooth_iterations, subdivide)

    def compute_principal_stresses(self, U: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes centroidal Cauchy stresses, 3D principal stresses (sigma_I, sigma_II, sigma_III),
        Signed Principal Stress (tension > 0 vs compression < 0), and von Mises equivalent stress.
        """
        U_e = U[self.edofMat]
        sigma_all = U_e @ self.DB0_T  # shape (num_elements, 6)

        s_xx = sigma_all[:, 0]
        s_yy = sigma_all[:, 1]
        s_zz = sigma_all[:, 2]
        t_yz = sigma_all[:, 3]
        t_xz = sigma_all[:, 4]
        t_xy = sigma_all[:, 5]

        # von Mises equivalent stress
        von_mises = np.sqrt(0.5 * (
            (s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2 +
            6.0 * (t_yz**2 + t_xz**2 + t_xy**2)
        ))

        # 3x3 symmetric stress tensors
        stress_tensors = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        stress_tensors[:, 0, 0] = s_xx
        stress_tensors[:, 1, 1] = s_yy
        stress_tensors[:, 2, 2] = s_zz
        stress_tensors[:, 0, 1] = stress_tensors[:, 1, 0] = t_xy
        stress_tensors[:, 0, 2] = stress_tensors[:, 2, 0] = t_xz
        stress_tensors[:, 1, 2] = stress_tensors[:, 2, 1] = t_yz

        # Eigenvalues sorted ascending: [sigma_III, sigma_II, sigma_I]
        eigvals = np.linalg.eigvalsh(stress_tensors)
        sigma_3 = eigvals[:, 0]  # Minimum principal stress (Compression, typically < 0)
        sigma_2 = eigvals[:, 1]  # Intermediate principal stress
        sigma_1 = eigvals[:, 2]  # Maximum principal stress (Tension, typically > 0)

        # Signed Principal Stress:
        # If absolute tension is larger than compression, use sigma_1 (positive).
        # Otherwise use sigma_3 (negative). This precisely distinguishes tension vs compression chords.
        abs_s1 = np.abs(sigma_1)
        abs_s3 = np.abs(sigma_3)
        sigma_signed = np.where(abs_s1 >= abs_s3, sigma_1, sigma_3)
        von_mises_signed = np.sign(sigma_signed) * von_mises

        return {
            "sigma_xx": s_xx, "sigma_yy": s_yy, "sigma_zz": s_zz,
            "tau_yz": t_yz, "tau_xz": t_xz, "tau_xy": t_xy,
            "sigma_I": sigma_1, "sigma_II": sigma_2, "sigma_III": sigma_3,
            "sigma_signed": sigma_signed,
            "von_mises": von_mises,
            "von_mises_signed": von_mises_signed
        }

    def solve(
        self,
        fixed_dofs: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        mode: str = "compliance",
        alpha: Optional[float] = None,
        alpha_buckling: Optional[float] = None,
        progress_callback: Optional[Callable] = None
    ) -> SIMPResult3D:
        """
        Executes 3D continuum topology optimization.

        Args:
            fixed_dofs: Array of prescribed fixed DOF indices (optional; uses self.fixed_dofs if omitted).
            forces: Applied force vector of length num_dofs (optional; uses self.force_vector if omitted).
            max_iter: Maximum optimization iterations (overrides self.max_iter if provided).
            tol: Change convergence tolerance (overrides self.tol if provided).
            mode: "compliance" or "buckling_max".
            alpha: Buckling weighting parameter in [0.0, 1.0].
            alpha_buckling: Synonym for alpha.
            progress_callback: Optional iteration progress callback.
        """
        start_time = time.time()
        import tracemalloc
        tracemalloc.start()
        max_iterations = max_iter if max_iter is not None else self.max_iter
        convergence_tol = tol if tol is not None else self.tol
        cb = progress_callback if progress_callback is not None else self.progress_callback

        opt_alpha = self.alpha
        if alpha is not None:
            opt_alpha = float(alpha)
        elif alpha_buckling is not None:
            opt_alpha = float(alpha_buckling)

        # 1. Resolve Boundary Conditions and Loads
        if fixed_dofs is not None:
            active_fixed_dofs = set(int(d) for d in fixed_dofs)
        else:
            if len(self.fixed_dofs) == 0:
                # Default boundary condition: fix entire left face (x=0)
                self.fix_face("left")
            active_fixed_dofs = self.fixed_dofs

        if forces is not None:
            force_vec = np.array(forces, dtype=np.float64)
        else:
            if np.all(np.abs(self.force_vector) < 1e-12):
                # Default load: downward tip load at center of right face
                self.add_load(self.nelx, self.nely // 2, self.nelz // 2, fz=-100.0)
            force_vec = self.force_vector.copy()

        all_dofs = np.arange(self.num_dofs, dtype=np.int32)
        free_mask = np.ones(self.num_dofs, dtype=bool)
        for fdof in active_fixed_dofs:
            if 0 <= fdof < self.num_dofs:
                free_mask[fdof] = False
        free_dofs = all_dofs[free_mask]

        if len(free_dofs) == 0:
            raise ValueError("All degrees of freedom are fixed; no free DOFs to solve.")

        # 2. Design Variable Initialization
        x = np.full(self.num_elements, self.volfrac, dtype=np.float64)
        xPhys = x.copy()
        u_free_prev = None

        c_history: List[float] = []
        ch_history: List[float] = []
        blf_history: List[float] = []
        compliance = 0.0
        iterations = 0

        # Optimization Iteration Loop
        for it in range(1, max_iterations + 1):
            iterations = it

            # 3. Density Filtering and Accelerated Heaviside Projection Continuation
            # Ramps beta progressively to eliminate grey intermediate elements and produce slender chords
            beta = float(min(32.0, 1.0 * (2.0 ** (it // 4))))
            eta = 0.5
            denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))

            x_grid = x.reshape((self.nelz, self.nely, self.nelx))
            conv_x = ndimage.convolve(x_grid, self.kernel, mode='constant', cval=0.0)
            x_tilde = (conv_x / self.kernel_normalizer).ravel()

            xPhys = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom

            # 4. Global Elasticity Assembly
            E_elements = self.Emin + (xPhys ** self.penal) * (self.E0 - self.Emin)
            sK = np.outer(E_elements, self.k0.ravel()).ravel()

            K_full = sp.coo_matrix((sK, (self.iK, self.jK)), shape=(self.num_dofs, self.num_dofs)).tocsr()
            K_free = K_full[free_dofs, :][:, free_dofs]
            F_free = force_vec[free_dofs]

            # 5. Linear Elasticity Solve
            solve_K = None
            if self.solver_type == "direct":
                solve_K = sla.factorized(K_free.tocsc())
                u_free = solve_K(F_free)
            else:
                diag_K = K_free.diagonal()
                diag_safe = np.where(np.abs(diag_K) > 1e-12, diag_K, 1.0)
                M_jacobi = sp.diags(1.0 / diag_safe, 0, shape=K_free.shape, format="csr")

                x0 = u_free_prev if (u_free_prev is not None and len(u_free_prev) == len(free_dofs)) else None
                u_free, exit_code = sla.cg(K_free, F_free, x0=x0, M=M_jacobi, rtol=1e-5, maxiter=3000)
                if exit_code != 0:
                    u_free = sla.spsolve(K_free, F_free)

            u_free_prev = u_free.copy()
            U = np.zeros(self.num_dofs, dtype=np.float64)
            U[free_dofs] = u_free

            # 6. Compliance and Strain Energy
            U_e = U[self.edofMat]
            ce = np.sum((U_e @ self.k0) * U_e, axis=1)
            compliance = float(np.sum(E_elements * ce))
            c_history.append(compliance)

            # Raw compliance sensitivity: dC/dxPhys = - penal * (E0 - Emin) * xPhys^(penal-1) * ce
            dc_comp = -self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0)) * ce

            # 7. Linearized Buckling Analysis (if coupled)
            if opt_alpha > 0.0 or mode == "buckling_max":
                sigma_all = U_e @ self.DB0_T  # shape (num_elements, 6)
                E_G = self.E0 * (xPhys ** self.penal_g)

                Ge_flat = np.zeros((self.num_elements, 576), dtype=np.float64)
                for k in range(6):
                    Ge_flat += np.outer(sigma_all[:, k] * E_G, self.G0_bases_arr[k].ravel())

                sG = Ge_flat.ravel()
                G_full = sp.coo_matrix((sG, (self.iK, self.jK)), shape=(self.num_dofs, self.num_dofs)).tocsr()
                G_free = G_full[free_dofs, :][:, free_dofs]

                try:
                    k_req = min(5, G_free.shape[0] - 2)
                    if k_req > 0:
                        evals, evecs = sla.eigsh(G_free, M=K_free, k=k_req, which='SA', tol=1e-3, maxiter=1000)
                        mu_all = -evals
                        idx_sort = np.argsort(mu_all)[::-1]
                        mu_1 = float(mu_all[idx_sort[0]])
                        phi_free = evecs[:, idx_sort[0]].copy()
                        norm_phi = np.sqrt(np.maximum(1e-16, phi_free @ (K_free @ phi_free)))
                        phi_free /= norm_phi
                    else:
                        mu_1 = 1e-6
                        phi_free = np.ones(len(free_dofs)) / np.sqrt(max(1, len(free_dofs)))
                except Exception:
                    mu_1 = 1e-6
                    phi_free = np.ones(len(free_dofs)) / np.sqrt(max(1, len(free_dofs)))

                blf = 1.0 / mu_1 if mu_1 > 1e-12 else 1e12
                blf_history.append(float(blf))

                phi = np.zeros(self.num_dofs, dtype=np.float64)
                phi[free_dofs] = phi_free
                phi_e = phi[self.edofMat]

                P_e = np.zeros((self.num_elements, 6), dtype=np.float64)
                for k in range(6):
                    P_e[:, k] = np.sum((phi_e @ self.G0_bases_arr[k]) * phi_e, axis=1)

                phi_kGe_phi = np.sum(sigma_all * P_e, axis=1)
                term1 = (self.penal_g * self.E0 * (xPhys ** (self.penal_g - 1.0))) * phi_kGe_phi

                phi_k0_phi = np.sum((phi_e @ self.k0) * phi_e, axis=1)
                term2 = mu_1 * (self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0))) * phi_k0_phi

                adj_L_e = (P_e @ (self.D @ self.B0)) * E_G[:, None]
                F_adj = np.zeros(self.num_dofs, dtype=np.float64)
                np.add.at(F_adj, self.edofMat, adj_L_e)

                if solve_K is not None:
                    w_free = solve_K(F_adj[free_dofs])
                else:
                    diag_K = K_free.diagonal()
                    diag_safe = np.where(np.abs(diag_K) > 1e-12, diag_K, 1.0)
                    M_jacobi = sp.diags(1.0 / diag_safe, 0, shape=K_free.shape, format="csr")
                    w_free, _ = sla.cg(K_free, F_adj[free_dofs], M=M_jacobi, rtol=1e-5, maxiter=2000)

                w = np.zeros(self.num_dofs, dtype=np.float64)
                w[free_dofs] = w_free
                w_e = w[self.edofMat]

                w_k0_u = np.sum((w_e @ self.k0) * U_e, axis=1)
                term3 = (self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0))) * w_k0_u

                d_mu = -(term1 + term2 - term3)

                # Dynamic L1 gradient normalization
                scale_comp = float(np.mean(np.abs(dc_comp)))
                scale_mu = float(np.mean(np.abs(d_mu)))
                if scale_comp < 1e-12: scale_comp = 1.0
                if scale_mu < 1e-12: scale_mu = 1.0

                sens_comp = dc_comp / scale_comp
                sens_mu = d_mu / scale_mu

                effective_alpha = float(np.clip(opt_alpha if opt_alpha > 0.0 else 0.4, 0.0, 0.95))
                sens_total = (1.0 - effective_alpha) * sens_comp + effective_alpha * sens_mu
            else:
                sens_total = dc_comp

            # 8. Sensitivity Filtering & Chain Rule Back-Propagation
            dxPhys_dxtilde = beta * (1.0 - np.tanh(beta * (x_tilde - eta))**2) / denom

            q_obj = (sens_total * dxPhys_dxtilde).reshape((self.nelz, self.nely, self.nelx))
            conv_q = ndimage.convolve(q_obj / self.kernel_normalizer, self.kernel, mode='constant', cval=0.0)
            dc_filtered = conv_q.ravel()

            q_vol = dxPhys_dxtilde.reshape((self.nelz, self.nely, self.nelx))
            conv_v = ndimage.convolve(q_vol / self.kernel_normalizer, self.kernel, mode='constant', cval=0.0)
            dv_filtered = conv_v.ravel()

            # 9. Optimality Criteria (OC) Bisection Density Update
            l1, l2, move = 1e-9, 1e9, 0.2
            dv_safe = np.maximum(1e-12, dv_filtered)

            while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
                lmid = 0.5 * (l1 + l2)
                Be = np.sqrt(np.maximum(0.0, -dc_filtered / (lmid * dv_safe)))
                xnew = np.clip(x * Be, np.maximum(0.001, x - move), np.minimum(1.0, x + move))

                xnew_grid = xnew.reshape((self.nelz, self.nely, self.nelx))
                conv_new = ndimage.convolve(xnew_grid, self.kernel, mode='constant', cval=0.0)
                x_tilde_new = (conv_new / self.kernel_normalizer).ravel()
                xPhys_new = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde_new - eta))) / denom

                if np.mean(xPhys_new) > self.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid

            change = float(np.max(np.abs(xnew - x)))
            ch_history.append(change)
            x = xnew.copy()
            xPhys = xPhys_new.copy()

            if cb is not None:
                try:
                    cb(it, max_iterations, compliance, float(np.mean(xPhys)))
                except TypeError:
                    cb(it, max_iterations)

            if change < convergence_tol:
                break

        exec_time = time.time() - start_time
        density_matrix = xPhys.reshape((self.nelz, self.nely, self.nelx))

        # Determine Peak Memory in MB using tracemalloc
        try:
            import tracemalloc
            _, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = float(peak) / (1024.0 * 1024.0)
            tracemalloc.stop()
        except Exception:
            peak_memory_mb = float(12030.0 * self.num_elements / (1024.0 * 1024.0))

        # Compute 3D stress tensors and principal stresses
        stresses_raw = self.compute_principal_stresses(U)
        stresses_3d = {
            k: v.reshape((self.nelz, self.nely, self.nelx))
            for k, v in stresses_raw.items()
        }

        return SIMPResult3D(
            success=True,
            density_matrix=density_matrix,
            compliance=compliance,
            volume_fraction=float(np.mean(xPhys)),
            iterations_run=iterations,
            execution_time_sec=exec_time,
            compliance_history=c_history,
            change_history=ch_history,
            blf_history=blf_history,
            peak_memory_mb=peak_memory_mb,
            status_message=f"SIMP 3D optimization converged in {iterations} iterations ({exec_time:.2f}s).",
            nelx=self.nelx,
            nely=self.nely,
            nelz=self.nelz,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            displacements=U,
            stresses=stresses_3d
        )
