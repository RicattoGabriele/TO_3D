"""
2D SIMP Continuum Topology Optimization Engine.
Physics-driven material distribution optimizer using Plane Stress Q4 finite elements,
SIMP material interpolation with cubic penalization, spatial sensitivity filtering (Helmholtz-type),
and deterministic Optimality Criteria (OC) update loop.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.optimize




def oc_update_mma(it, x, dg0, g1, dg1, ocPar, xOld, xOld1, as_prev, beta=2.0, restartAsy=False):
    move, asReduce, asRelax = ocPar
    xU = np.minimum(x + move, 1.0)
    xL = np.maximum(x - move, 0.0)
    if it < 2 or restartAsy:
        as_new = np.zeros((len(x), 2))
        as_new[:, 0] = x - 0.5 * (xU - xL) / (beta + 1)
        as_new[:, 1] = x + 0.5 * (xU - xL) / (beta + 1)
    else:
        tmp = (x - xOld) * (xOld - xOld1)
        gm = np.ones(len(x))
        gm[tmp > 0] = asRelax
        gm[tmp < 0] = asReduce
        as_new = np.zeros((len(x), 2))
        as_new[:, 0] = x - gm * (xOld - as_prev[:, 0])
        as_new[:, 1] = x + gm * (as_prev[:, 1] - xOld)
        
    xL = np.maximum(0.9 * as_new[:, 0] + 0.1 * x, xL)
    xU = np.minimum(0.9 * as_new[:, 1] + 0.1 * x, xU)
    
    p0_0, q0_0 = np.maximum(dg0, 0) * dg0, np.maximum(-dg0, 0) * -dg0
    p1_0, q1_0 = np.maximum(dg1, 0) * dg1, np.maximum(-dg1, 0) * -dg1
    
    p0 = p0_0 * (as_new[:, 1] - x)**2
    q0 = q0_0 * (x - as_new[:, 0])**2
    p1 = p1_0 * (as_new[:, 1] - x)**2
    q1 = q1_0 * (x - as_new[:, 0])**2
    
    def primalProj(lm):
        num = np.sqrt(p0 + lm * p1) * as_new[:, 0] + np.sqrt(q0 + lm * q1) * as_new[:, 1]
        den = np.sqrt(p0 + lm * p1) + np.sqrt(q0 + lm * q1) + 1e-16
        return np.clip(num / den, xL, xU)
        
    def psiDual(lm):
        x_lm = primalProj(lm)
        t1 = g1 - np.dot(as_new[:, 1] - x, p1_0) - np.dot(x - as_new[:, 0], q1_0)
        t2 = np.sum(p1 / np.maximum(as_new[:, 1] - x_lm, 1e-12))
        t3 = np.sum(q1 / np.maximum(x_lm - as_new[:, 0], 1e-12))
        return t1 + t2 + t3

    lmUp = 1e6
    ps0, psUp = psiDual(0.0), psiDual(lmUp)
    if ps0 * psUp < 0:
        try: lmid = scipy.optimize.brentq(psiDual, 0.0, lmUp)
        except ValueError: lmid = 0.0
        x_new = primalProj(lmid)
    elif ps0 <= 0:
        lmid, x_new = 0.0, primalProj(0.0)
    else:
        lmid, x_new = lmUp, primalProj(lmUp)
    return x_new, as_new, lmid

@dataclass
class SIMPResult:
    """Result of 2D SIMP topology optimization."""
    success: bool
    status_message: str
    density_matrix: np.ndarray  # 2D array of shape (nely, nelx) with values in [0, 1]
    compliance: float
    volume_fraction: float
    iterations_run: int
    execution_time_sec: float
    compliance_history: List[float]
    change_history: List[float]
    nelx: int
    nely: int
    dx: float
    dy: float
    blf_history: Optional[List[float]] = None


class SIMPOptimizer2D:
    """
    Solves minimum compliance topology optimization on a 2D continuum domain.
    """

    def __init__(
        self,
        nelx: int = 60,
        nely: int = 30,
        dx: float = 1.0,
        dy: float = 1.0,
        thickness: float = 1.0,
        E0: float = 1.0,
        Emin: float = 1e-9,
        nu: float = 0.3,
        penal: float = 3.0,
        rmin: float = 2.5,
        volfrac: float = 0.4,
    ):
        """
        Args:
            nelx: Number of elements in X direction
            nely: Number of elements in Y direction
            dx: Element size in X (mm)
            dy: Element size in Y (mm)
            thickness: Out-of-plane thickness (mm)
            E0: Young's modulus of solid material (MPa)
            Emin: Minimum stiffness for void elements (avoids singularity)
            nu: Poisson's ratio
            penal: SIMP penalization exponent (typically 3.0)
            rmin: Sensitivity filter radius (in element units or physical mm)
            volfrac: Target volume fraction in [0.05, 0.95]
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.dx = float(dx)
        self.dy = float(dy)
        self.thickness = float(thickness)
        self.E0 = float(E0)
        self.Emin = float(Emin)
        self.nu = float(nu)
        self.penal = float(penal)
        self.penal_g = 6.0
        self.rmin = float(rmin)
        self.volfrac = float(volfrac)

        self.num_elements = self.nelx * self.nely
        self.num_nodes = (self.nelx + 1) * (self.nely + 1)
        self.num_dofs = 2 * self.num_nodes

        # Element stiffness matrix k0
        self.k0 = self._build_element_stiffness()
        self._init_buckling_matrices()

        # Assembly indexing vectors
        self.edofMat = self._build_edof_matrix()

        # Spatial filter weights
        self.H, self.Hs = self._build_spatial_filter()

        # Boundary conditions
        self.fixed_dofs: Set[int] = set()
        self.force_vector = np.zeros(self.num_dofs, dtype=np.float64)

    def _build_element_stiffness(self) -> np.ndarray:
        """
        Exact Q4 plane stress element stiffness matrix using 2x2 Gauss quadrature.
        Returns 8x8 symmetric matrix.
        """
        E = self.E0
        nu = self.nu
        t = self.thickness
        dx = self.dx
        dy = self.dy

        # Plane stress constitutive matrix D
        factor = E / (1.0 - nu**2)
        D = factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ], dtype=np.float64)

        k = np.zeros((8, 8), dtype=np.float64)
        gauss_pts = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
        det_J = (dx * dy) / 4.0

        for xi in gauss_pts:
            for eta in gauss_pts:
                # Shape function derivatives w.r.t xi, eta
                dNdxi = np.array([
                    -0.25 * (1.0 - eta),
                     0.25 * (1.0 - eta),
                     0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta)
                ])
                dNdeta = np.array([
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                     0.25 * (1.0 + xi),
                     0.25 * (1.0 - xi)
                ])

                # Derivatives w.r.t physical x, y
                dNx = dNdxi * (2.0 / dx)
                dNy = dNdeta * (2.0 / dy)

                # Strain-displacement matrix B (3x8)
                # DOFs ordered: [u1, v1, u2, v2, u3, v3, u4, v4]
                B = np.zeros((3, 8), dtype=np.float64)
                for i in range(4):
                    B[0, 2 * i] = dNx[i]
                    B[1, 2 * i + 1] = dNy[i]
                    B[2, 2 * i] = dNy[i]
                    B[2, 2 * i + 1] = dNx[i]

                k += (B.T @ D @ B) * det_J * t

        return k


    def _init_buckling_matrices(self):
        nu, E0, dx, dy, t = self.nu, self.E0, self.dx, self.dy, self.thickness
        factor = E0 / (1.0 - nu**2)
        self.Cmat0 = factor * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]], dtype=np.float64)
        
        dNdxi = np.array([-0.25, 0.25, 0.25, -0.25])
        dNdeta = np.array([-0.25, -0.25, 0.25, 0.25])
        dNx, dNy = dNdxi * (2.0 / dx), dNdeta * (2.0 / dy)
        
        self.B0 = np.zeros((3, 8), dtype=np.float64)
        for i in range(4):
            self.B0[0, 2*i], self.B0[1, 2*i+1] = dNx[i], dNy[i]
            self.B0[2, 2*i], self.B0[2, 2*i+1] = dNy[i], dNx[i]
            
        B1 = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            B1[0, 2*i], B1[1, 2*i+1], B1[2, 2*i], B1[3, 2*i+1] = dNx[i], dNx[i], dNy[i], dNy[i]
            
        Tx_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
        Ty_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        Txy_mat = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        
        area = dx * dy * t
        self.Tx = (B1.T @ Tx_mat @ B1) * area
        self.Ty = (B1.T @ Ty_mat @ B1) * area
        self.Txy = (B1.T @ Txy_mat @ B1) * area

    def _build_edof_matrix(self) -> np.ndarray:
        """
        Builds matrix of element DOFs of size (num_elements, 8).
        Node numbering: node (i, j) where i in [0..nelx], j in [0..nely].
        Node ID = i * (nely + 1) + j.
        Element ID = elx * nely + ely.
        """
        edofMat = np.zeros((self.num_elements, 8), dtype=np.int32)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el_idx = elx * self.nely + ely
                # 4 corner nodes of element (elx, ely):
                # n1: bottom-left (elx, ely)
                # n2: bottom-right (elx+1, ely)
                # n3: top-right (elx+1, ely+1)
                # n4: top-left (elx, ely+1)
                n1 = elx * (self.nely + 1) + ely
                n2 = (elx + 1) * (self.nely + 1) + ely
                n3 = (elx + 1) * (self.nely + 1) + (ely + 1)
                n4 = elx * (self.nely + 1) + (ely + 1)

                edofMat[el_idx, :] = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n3, 2 * n3 + 1,
                    2 * n4, 2 * n4 + 1
                ]
        return edofMat

    def _build_spatial_filter(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Builds spatial sensitivity filter matrix H (sparse) to eliminate checkerboards
        and enforce minimum length scale rmin.
        """
        rmin = self.rmin
        nely = self.nely
        nelx = self.nelx
        num_el = self.num_elements

        iH = []
        jH = []
        sH = []

        # Loop over elements and neighbor box of radius ceil(rmin)
        r_ceil = int(math.ceil(rmin))
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = i1 * nely + j1
                imin = max(0, i1 - r_ceil)
                imax = min(nelx, i1 + r_ceil + 1)
                jmin = max(0, j1 - r_ceil)
                jmax = min(nely, j1 + r_ceil + 1)

                for i2 in range(imin, imax):
                    for j2 in range(jmin, jmax):
                        e2 = i2 * nely + j2
                        dist = math.hypot((i1 - i2) * self.dx, (j1 - j2) * self.dy)
                        weight = max(0.0, rmin - dist)
                        if weight > 0:
                            iH.append(e1)
                            jH.append(e2)
                            sH.append(weight)

        H = sp.csr_matrix((sH, (iH, jH)), shape=(num_el, num_el), dtype=np.float64)
        Hs = np.array(H.sum(axis=1)).flatten()
        return H, Hs

    def node_id(self, i: int, j: int) -> int:
        """Returns node ID from grid coordinate (i in 0..nelx, j in 0..nely)."""
        if not (0 <= i <= self.nelx and 0 <= j <= self.nely):
            raise IndexError(f"Node indices ({i}, {j}) out of range [0..{self.nelx}, 0..{self.nely}]")
        return i * (self.nely + 1) + j

    def clear_boundary_conditions(self):
        """Clears all fixed DOFs and applied forces."""
        self.fixed_dofs.clear()
        self.force_vector.fill(0.0)

    def fix_node(self, i: int, j: int, fix_x: bool = True, fix_y: bool = True):
        """Fix displacements of node (i, j)."""
        nid = self.node_id(i, j)
        if fix_x:
            self.fixed_dofs.add(2 * nid)
        if fix_y:
            self.fixed_dofs.add(2 * nid + 1)

    def fix_wall(self, side: str = "left", fix_x: bool = True, fix_y: bool = True):
        """Fix an entire boundary wall ('left', 'right', 'bottom', 'top')."""
        if side == "left":
            for j in range(self.nely + 1):
                self.fix_node(0, j, fix_x, fix_y)
        elif side == "right":
            for j in range(self.nely + 1):
                self.fix_node(self.nelx, j, fix_x, fix_y)
        elif side == "bottom":
            for i in range(self.nelx + 1):
                self.fix_node(i, 0, fix_x, fix_y)
        elif side == "top":
            for i in range(self.nelx + 1):
                self.fix_node(i, self.nely, fix_x, fix_y)

    def add_load(self, i: int, j: int, fx: float = 0.0, fy: float = 0.0):
        """Apply concentrated point load in Newtons to node (i, j)."""
        nid = self.node_id(i, j)
        self.force_vector[2 * nid] += fx
        self.force_vector[2 * nid + 1] += fy

    def solve(
        self,
        max_iter: int = 45,
        tol: float = 0.01,
        mode: str = "compliance",
        alpha_buckling: float = 0.40,
        nEig: int = 1,
        ks_rho: float = 40.0,
        progress_callback = None
    ) -> SIMPResult:
        """
        Runs topology optimization.
        
        Args:
            max_iter: Maximum number of iterations (default: 45)
            tol: Convergence tolerance on change in design variables (default: 0.01)
            mode: "compliance" for standard minimum compliance (symmetric under force reversal),
                  "buckling_max" for compliance + linearized buckling resistance (asymmetric, reinforces compressed chords).
            alpha_buckling: Weight factor for buckling sensitivity in [0.0, 1.0] when mode='buckling_max' (default: 0.40).
        """
        start_time = time.time()
        if len(self.fixed_dofs) < 3:
            raise ValueError("Insufficient boundary conditions: need at least 3 fixed DOFs to prevent rigid body motion.")
        if np.all(np.abs(self.force_vector) < 1e-12):
            raise ValueError("No external forces applied to domain.")

        x = np.full(self.num_elements, self.volfrac, dtype=np.float64)
        xPhys = x.copy()

        all_dofs = np.arange(self.num_dofs, dtype=np.int32)
        free_mask = np.ones(self.num_dofs, dtype=bool)
        for fdof in self.fixed_dofs:
            free_mask[fdof] = False
        free_dofs = all_dofs[free_mask]

        iK = np.kron(self.edofMat, np.ones((8, 1), dtype=np.int32)).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8), dtype=np.int32)).flatten()

        c_history, ch_history, blf_history = [], [], []
        compliance, iterations = 0.0, 0

        for it in range(1, max_iter + 1):
            iterations = it
            if progress_callback:
                progress_callback(it, max_iter)
                
            # 0. Density Filtering and Heaviside Projection
            # Continuation scheme: beta doubles every 15 iterations (max 16)
            beta = min(16.0, 1.0 * (2.0 ** (it // 15)))
            eta = 0.5
            
            x_tilde = np.array(self.H @ x).flatten() / self.Hs
            denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
            xPhys = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
            
            # 1. Linear Elasticity FEA
            E_elements = self.Emin + (xPhys ** self.penal) * (self.E0 - self.Emin)
            sK = (self.k0.flatten()[np.newaxis, :] * E_elements[:, np.newaxis]).flatten()

            K_full = sp.coo_matrix((sK, (iK, jK)), shape=(self.num_dofs, self.num_dofs)).tocsr()
            K_free = K_full[free_dofs, :][:, free_dofs]
            solve_K = sla.factorized(K_free.tocsc())
            F_free = self.force_vector[free_dofs]

            U_free = solve_K(F_free)
            U = np.zeros(self.num_dofs, dtype=np.float64)
            U[free_dofs] = U_free

            U_e = U[self.edofMat]
            ce = np.sum((U_e @ self.k0) * U_e, axis=1)
            compliance = float(np.sum(E_elements * ce))
            c_history.append(compliance)

            # Compliance sensitivity: dC/dx_e = - penal * (E0 - Emin) * x^(penal - 1) * ce
            dc_comp = -self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0)) * ce

            if mode == "compliance":
                sens_total = dc_comp
            else:
                # 2. Linearized Buckling Analysis (LBA) and Geometric Stiffness G
                sigma_e = (self.Cmat0 @ self.B0 @ U_e.T).T
                Ge = sigma_e[:, 0, None, None] * self.Tx + sigma_e[:, 1, None, None] * self.Ty + sigma_e[:, 2, None, None] * self.Txy
                
                # Stress stiffness penalization pG = 3.0 to prevent spurious void buckling
                E_G = (xPhys ** 3.0) * self.E0
                Ge_penalized = Ge * E_G[:, None, None]

                G_full = sp.coo_matrix((Ge_penalized.flatten(), (iK, jK)), shape=(self.num_dofs, self.num_dofs)).tocsr()
                G_free = G_full[free_dofs, :][:, free_dofs]

                try:
                    # Request nEig + 4 to ensure robustness for tight clusters
                    k_request = min(nEig + 4, G_free.shape[0] - 2)
                    evals, evecs = sla.eigsh(G_free, M=K_free, k=k_request, which='SA', tol=1e-3)
                    # We want the lowest buckling eigenvalues, which correspond to the HIGHEST negative evals
                    # eigs with 'SA' returns smallest algebraic, so most negative.
                    # evals are sorted algebraically. We want the nEig most negative ones.
                    # mu = -evals. We sort mu descending to pick the largest mu (lowest BLF).
                    mu_all = -evals
                    idx_sort = np.argsort(mu_all)[::-1]
                    mu_all = mu_all[idx_sort]
                    evecs = evecs[:, idx_sort]
                    
                    mu_active = mu_all[:nEig]
                    evecs_active = evecs[:, :nEig]
                    
                    for i in range(nEig):
                        norm_phi = np.sqrt(np.maximum(1e-16, evecs_active[:, i] @ (K_free @ evecs_active[:, i])))
                        evecs_active[:, i] /= norm_phi
                except Exception:
                    mu_active = np.array([1e-6] * nEig)
                    evecs_active = np.ones((len(free_dofs), nEig)) / np.sqrt(len(free_dofs))

                mu_max = np.max(mu_active)
                if nEig == 1:
                    mu_ks = mu_max
                    weights_ks = np.array([1.0])
                else:
                    sum_exp = np.sum(np.exp(ks_rho * (mu_active - mu_max)))
                    mu_ks = mu_max + (1.0 / ks_rho) * np.log(sum_exp)
                    weights_ks = np.exp(ks_rho * (mu_active - mu_max)) / sum_exp

                blf = 1.0 / mu_ks if mu_ks > 1e-12 else 1e12
                blf_history.append(float(blf))

                d_mu_ks = np.zeros(self.num_elements, dtype=np.float64)

                for idx_m in range(nEig):
                    phi = np.zeros(self.num_dofs, dtype=np.float64)
                    phi[free_dofs] = evecs_active[:, idx_m]
                    phi_e = phi[self.edofMat]
                    mu_m = mu_active[idx_m]

                    phi_Ge_phi = np.einsum('ni,nij,nj->n', phi_e, Ge, phi_e)
                    term1 = phi_Ge_phi * 3.0 * self.E0 * (xPhys ** 2.0)

                    phi_Ke_phi = np.einsum('ni,ij,nj->n', phi_e, self.k0, phi_e)
                    term2 = mu_m * phi_Ke_phi * self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0))

                    P = np.column_stack([
                        np.einsum('ni,ij,nj->n', phi_e, self.Tx, phi_e),
                        np.einsum('ni,ij,nj->n', phi_e, self.Ty, phi_e),
                        np.einsum('ni,ij,nj->n', phi_e, self.Txy, phi_e)
                    ])
                    adj_L_e = (P @ (self.Cmat0 @ self.B0)) * E_G[:, None]
                    adj_L = np.zeros(self.num_dofs, dtype=np.float64)
                    np.add.at(adj_L, self.edofMat, adj_L_e)

                    w_free = solve_K(adj_L[free_dofs])
                    w = np.zeros(self.num_dofs, dtype=np.float64)
                    w[free_dofs] = w_free
                    w_e = w[self.edofMat]

                    term3 = np.sum((w_e @ self.k0) * U_e, axis=1) * self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0))

                    d_mu_m = -(term1 + term2 - term3)
                    d_mu_ks += weights_ks[idx_m] * d_mu_m

                d_mu = d_mu_ks

                # Normalize scales
                scale_comp = float(np.mean(np.abs(dc_comp)))
                scale_mu = float(np.mean(np.abs(d_mu)))
                if scale_comp < 1e-12: scale_comp = 1.0
                if scale_mu < 1e-12: scale_mu = 1.0

                sens_comp = dc_comp / scale_comp
                sens_mu = d_mu / scale_mu

                alpha = float(np.clip(alpha_buckling, 0.0, 0.95))
                sens_total = (1.0 - alpha) * sens_comp + alpha * sens_mu

            # 3. Spatial Filtering Chain Rule (Density Filter + Heaviside)
            dxPhys_dxtilde = beta * (1.0 - np.tanh(beta * (x_tilde - eta))**2) / denom
            dc_filtered = np.array(self.H @ (sens_total * dxPhys_dxtilde / self.Hs)).flatten()

            # 4. Volume derivatives via Chain Rule
            dv_filtered = np.array(self.H @ (dxPhys_dxtilde / self.Hs)).flatten()

            # 4. Deterministic Optimality Criteria (OC) Density Update with Bisection
            l1, l2, move = 1e-9, 1e9, 0.2
            while (l2 - l1) / (l1 + l2) > 1e-4:
                lmid = 0.5 * (l2 + l1)
                Be = np.sqrt(np.maximum(0.0, -dc_filtered / (lmid * np.maximum(1e-12, dv_filtered))))
                xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * Be))))
                x_tilde_new = np.array(self.H @ xnew).flatten() / self.Hs
                xPhys_new = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde_new - eta))) / denom
                
                if np.mean(xPhys_new) > self.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid

            change = float(np.max(np.abs(xnew - x)))
            ch_history.append(change)
            x = xnew.copy()
            xPhys = xPhys_new.copy()

            if change < tol:
                break

        density_matrix = np.zeros((self.nely, self.nelx), dtype=np.float64)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                density_matrix[ely, elx] = xPhys[elx * self.nely + ely]

        exec_time = time.time() - start_time
        return SIMPResult(
            success=True,
            status_message=f"SIMP optimization converged in {iterations} iterations ({exec_time:.2f}s).",
            density_matrix=density_matrix,
            compliance=compliance,
            volume_fraction=float(np.mean(xPhys)),
            iterations_run=iterations,
            execution_time_sec=exec_time,
            compliance_history=c_history,
            change_history=ch_history,
            nelx=self.nelx,
            nely=self.nely,
            dx=self.dx,
            dy=self.dy,
            blf_history=blf_history if mode == "buckling_max" else None
        )

    def to_plotly_heatmap(
        self,
        res: SIMPResult,
        threshold: float = 0.5,
        colormap: str = "Jet_r",
        smooth: bool = False
    ):
        """
        Creates an interactive Plotly continuous density heatmap with isoline contours,
        support markers, and external load arrows.
        """
        import plotly.graph_objects as go

        x_coords = np.linspace(0, self.nelx * self.dx, self.nelx)
        y_coords = np.linspace(0, self.nely * self.dy, self.nely)

        fig = go.Figure()

        # 1. Continuous Density Heatmap
        fig.add_trace(go.Heatmap(
            z=res.density_matrix,
            x=x_coords,
            y=y_coords,
            colorscale=colormap,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="Relative Density ρ", font=dict(color="#f8fafc", size=11)),
                tickfont=dict(color="#94a3b8", size=10),
                len=0.85
            ),
            hoverongaps=False,
            hovertemplate="X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Density: %{z:.2f}<extra></extra>"
        ))

        # 2. Contour line at threshold (Solid Boundary)
        fig.add_trace(go.Contour(
            z=res.density_matrix,
            x=x_coords,
            y=y_coords,
            contours=dict(
                start=threshold,
                end=threshold,
                size=1.0,
                coloring="none"
            ),
            line=dict(color="#ffffff", width=2.5, dash="solid"),
            showscale=False,
            hoverinfo="none",
            name=f"Solid Boundary (ρ = {threshold:.2f})"
        ))

        # 3. Add Boundary Conditions
        fixed_x, fixed_y, fixed_labels = [], [], []
        loaded_x, loaded_y, loaded_labels = [], [], []

        for nid in range(self.num_nodes):
            ix = nid // (self.nely + 1)
            iy = nid % (self.nely + 1)
            x_pos = ix * self.dx
            y_pos = iy * self.dy

            is_fixed_x = (2 * nid in self.fixed_dofs)
            is_fixed_y = (2 * nid + 1 in self.fixed_dofs)
            fx = self.force_vector[2 * nid]
            fy = self.force_vector[2 * nid + 1]

            if is_fixed_x or is_fixed_y:
                fixed_x.append(x_pos)
                fixed_y.append(y_pos)
                fixed_labels.append(f"Fixed ({ix}, {iy}) a ({x_pos:.0f}, {y_pos:.0f}) mm")

            if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                loaded_x.append(x_pos)
                loaded_y.append(y_pos)
                loaded_labels.append(f"Load ({ix}, {iy}): F=({fx:.0f}, {fy:.0f}) N")

                # Arrow annotation
                mag = math.hypot(fx, fy)
                ax = 0.0 if abs(fx) < 1e-6 else (-35.0 * fx / mag)
                ay = 0.0 if abs(fy) < 1e-6 else (35.0 * fy / mag)

                fig.add_annotation(
                    x=x_pos, y=y_pos,
                    ax=ax, ay=ay,
                    xref="x", yref="y",
                    axref="pixel", ayref="pixel",
                    text=f"F=({fx:.0f}, {fy:.0f})N",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="#ffffff",
                    font=dict(color="#ffffff", size=10, family="monospace")
                )

        # Support markers
        if fixed_x:
            fig.add_trace(go.Scatter(
                x=fixed_x, y=fixed_y,
                mode="markers",
                marker=dict(size=9, symbol="square", color="#333333", line=dict(color="#78350f", width=1.5)),
                hoverinfo="text",
                hovertext=fixed_labels,
                name="Fixed Support"
            ))

        # Load markers
        if loaded_x:
            fig.add_trace(go.Scatter(
                x=loaded_x, y=loaded_y,
                mode="markers",
                marker=dict(size=10, symbol="circle", color="#ef4444", line=dict(color="#ffffff", width=1.5)),
                hoverinfo="text",
                hovertext=loaded_labels,
                name="Applied Load"
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#020617",
            xaxis=dict(
                title="X (mm)",
                scaleanchor="y",
                scaleratio=1,
                gridcolor="#334155",
                zerolinecolor="#334155"
            ),
            yaxis=dict(
                title="Y (mm)",
                gridcolor="#334155",
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
            height=430
        )

        return fig


    def compute_principal_stresses(self, res: SIMPResult) -> Dict[str, np.ndarray]:
        """
        Computes 2D elemental stresses, principal stresses (sigma_I, sigma_II),
        and von Mises stress for the converged topology.
        
        Returns:
            Dict containing:
                'sigma_I': 2D array (nely, nelx) of maximum principal stress (tensile if > 0)
                'sigma_II': 2D array (nely, nelx) of minimum principal stress (compressive if < 0)
                'von_mises': 2D array (nely, nelx) of von Mises stress
                'sx', 'sy', 'sxy': 2D arrays of normal and shear stresses
        """
        all_dofs = np.arange(self.num_dofs, dtype=np.int32)
        free_mask = np.ones(self.num_dofs, dtype=bool)
        for fdof in self.fixed_dofs:
            free_mask[fdof] = False
        free_dofs = all_dofs[free_mask]

        E_elements = self.Emin + (res.density_matrix.T.flatten() ** self.penal) * (self.E0 - self.Emin)
        sK = (self.k0.flatten()[np.newaxis, :] * E_elements[:, np.newaxis]).flatten()
        iK = np.kron(self.edofMat, np.ones((8, 1), dtype=np.int32)).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8), dtype=np.int32)).flatten()

        K_full = sp.coo_matrix((sK, (iK, jK)), shape=(self.num_dofs, self.num_dofs)).tocsr()
        K_free = K_full[free_dofs, :][:, free_dofs]
        F_free = self.force_vector[free_dofs]

        U_free = sla.spsolve(K_free, F_free)
        U = np.zeros(self.num_dofs, dtype=np.float64)
        U[free_dofs] = U_free
        U_e = U[self.edofMat]

        sigma_e = (self.Cmat0 @ self.B0 @ U_e.T).T
        sx = sigma_e[:, 0]
        sy = sigma_e[:, 1]
        sxy = sigma_e[:, 2]

        s_avg = (sx + sy) / 2.0
        s_diff = np.sqrt(np.maximum(0.0, ((sx - sy) / 2.0)**2 + sxy**2))
        s_I = s_avg + s_diff
        s_II = s_avg - s_diff
        s_vm = np.sqrt(np.maximum(0.0, sx**2 - sx*sy + sy**2 + 3.0*(sxy**2)))

        s_signed = np.where(np.abs(s_I) > np.abs(s_II), s_I, s_II)
        return {
            "sigma_signed": s_signed.reshape((self.nelx, self.nely)).T,
            "sigma_I": s_I.reshape((self.nelx, self.nely)).T,
            "sigma_II": s_II.reshape((self.nelx, self.nely)).T,
            "von_mises": s_vm.reshape((self.nelx, self.nely)).T,
            "sx": sx.reshape((self.nelx, self.nely)).T,
            "sy": sy.reshape((self.nelx, self.nely)).T,
            "sxy": sxy.reshape((self.nelx, self.nely)).T
        }

    def to_scientific_figures(
        self,
        res: SIMPResult,
        threshold: float = 0.40,
        view_style: str = "dual_scientific",
        colormap: str = "Jet_r",
        smooth: bool = False
    ):
        """
        Creates publication-quality, high-contrast figures mirroring the state-of-the-art
        visualizations from Ferrari, Sigmund & Guest (2021).
        
        Args:
            res: SIMPResult object
            threshold: Material threshold to isolate solid structure (default: 0.40)
            view_style: "dual_scientific" (B&W + Stress side by side),
                        "silhouette_bw" (Crisp B&W solid geometry),
                        "stress_masked" (Compressive stress sigma_II with void blanking),
                        "density" (Classic continuous thermographic heatmap)
            colormap: Colormap for stress map (default: "Jet" or "Turbo")
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        x_coords = np.linspace(0, self.nelx * self.dx, self.nelx)
        y_coords = np.linspace(0, self.nely * self.dy, self.nely)

        # 1. Fallback to standard density heatmap
        if view_style == "density":
            return self.to_plotly_heatmap(res, threshold=threshold, colormap=colormap)

        stresses = self.compute_principal_stresses(res)
        s_signed = stresses["sigma_signed"]

        # Mask void elements below threshold as NaN so Plotly renders them transparent/white
        s_masked = np.where(res.density_matrix >= threshold, s_signed, np.nan)
        solid_binary = np.where(res.density_matrix >= threshold, 1.0, 0.0)

        valid_s = s_masked[~np.isnan(s_masked)]
        if len(valid_s) > 0:
            max_abs = max(abs(float(np.percentile(valid_s, 2))), abs(float(np.percentile(valid_s, 98))))
            min_val, max_val = -max_abs, max_abs
        else:
            min_val, max_val = -1.0, 1.0
        
        
        s_II_masked = s_masked  # Keep variable name for rest of code

        # Boundary conditions markers
        fixed_x, fixed_y = [], []
        loaded_x, loaded_y, loaded_arrows = [], [], []

        for nid in range(self.num_nodes):
            ix = nid // (self.nely + 1)
            iy = nid % (self.nely + 1)
            x_pos, y_pos = ix * self.dx, iy * self.dy
            if (2 * nid in self.fixed_dofs) or (2 * nid + 1 in self.fixed_dofs):
                fixed_x.append(x_pos)
                fixed_y.append(y_pos)
            fx = self.force_vector[2 * nid]
            fy = self.force_vector[2 * nid + 1]
            if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                loaded_x.append(x_pos)
                loaded_y.append(y_pos)
                mag = math.hypot(fx, fy)
                ax = 0.0 if abs(fx) < 1e-6 else (-35.0 * fx / mag)
                ay = 0.0 if abs(fy) < 1e-6 else (35.0 * fy / mag)
                loaded_arrows.append((x_pos, y_pos, ax, ay, f"F=({fx:.0f}, {fy:.0f})N"))

        if view_style == "dual_scientific":
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "<b>Optimized Topology (Solid Silhouette)</b>",
                    "<b>Signed Principal Stress Distribution</b>"
                ),
                horizontal_spacing=0.08
            )

            # Left: B&W Silhouette
            fig.add_trace(
                go.Heatmap(
                    z=solid_binary,
                    x=x_coords,
                    y=y_coords,
                    colorscale=[[0, "#ffffff"], [1, "#090d16"]],
                    showscale=False,
                    hoverinfo="none"
                ),
                row=1, col=1
            )
            # Smooth contour line on left
            fig.add_trace(
                go.Contour(
                    z=res.density_matrix,
                    x=x_coords,
                    y=y_coords,
                    contours=dict(start=threshold, end=threshold, size=1.0, coloring="none"),
                    line=dict(color="#090d16", width=2.0),
                    showscale=False,
                    hoverinfo="none"
                ),
                row=1, col=1
            )

            # Right: Masked Compressive Stress sigma_II
            fig.add_trace(
                go.Heatmap(
                    z=s_II_masked,
                    x=x_coords,
                    y=y_coords,
                    colorscale=colormap,
                    zmin=min_val,
                    zmax=max_val,
                    colorbar=dict(
                        title=dict(text="Principal Stress [MPa]", font=dict(color="#f8fafc", size=11)),
                        tickfont=dict(color="#94a3b8", size=10),
                        len=0.85,
                        x=1.02
                    ),
                    hoverongaps=False,
                    hovertemplate="X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>&sigma;<sub>1,2</sub>: %{z:.2e}<extra></extra>"
                ),
                row=1, col=2
            )
            # Contour line on right
            fig.add_trace(
                go.Contour(
                    z=res.density_matrix,
                    x=x_coords,
                    y=y_coords,
                    contours=dict(start=threshold, end=threshold, size=1.0, coloring="none"),
                    line=dict(color="#0f172a", width=1.5),
                    showscale=False,
                    hoverinfo="none"
                ),
                row=1, col=2
            )

            # Add markers
            if fixed_x:
                fig.add_trace(go.Scatter(x=fixed_x, y=fixed_y, mode="markers", marker=dict(size=7, symbol="square", color="#333333"), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=fixed_x, y=fixed_y, mode="markers", marker=dict(size=7, symbol="square", color="#333333"), showlegend=False), row=1, col=2)
            if loaded_x:
                fig.add_trace(go.Scatter(x=loaded_x, y=loaded_y, mode="markers", marker=dict(size=8, symbol="circle", color="#ef4444"), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=loaded_x, y=loaded_y, mode="markers", marker=dict(size=8, symbol="circle", color="#ef4444"), showlegend=False), row=1, col=2)

            for xp, yp, ax, ay, txt in loaded_arrows:
                fig.add_annotation(x=xp, y=yp, ax=ax, ay=ay, xref="x1", yref="y1", axref="pixel", ayref="pixel", text=txt, showarrow=True, arrowhead=2, arrowcolor="#ef4444", font=dict(color="#ef4444", size=14, family="Courier Prime"))
                fig.add_annotation(x=xp, y=yp, ax=ax, ay=ay, xref="x2", yref="y2", axref="pixel", ayref="pixel", text=txt, showarrow=True, arrowhead=2, arrowcolor="#ef4444", font=dict(color="#ef4444", size=14, family="Courier Prime"))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#ffffff",
                height=450,
                margin=dict(l=20, r=40, t=50, b=20)
            )
            fig.update_xaxes(scaleanchor="y", scaleratio=1, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title_text="X (mm)")
            fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title_text="Y (mm)")
            return fig

        elif view_style == "silhouette_bw":
            fig = go.Figure()
            if smooth:
                fig.add_trace(go.Contour(
                    z=res.density_matrix,
                    x=x_coords,
                    y=y_coords,
                    contours=dict(start=threshold, end=1.0, size=2.0, coloring="fill"),
                    colorscale=[[0, "#ffffff"], [1, "#020617"]],
                    line_width=0,
                    showscale=False,
                    hoverinfo="none",
                    name="Structure"
                ))
            else:
                fig.add_trace(go.Heatmap(
                    z=solid_binary,
                    x=x_coords,
                    y=y_coords,
                    colorscale=[[0, "#ffffff"], [1, "#020617"]],
                    showscale=False,
                    hoverinfo="none",
                    name="Structure"
                ))
            if fixed_x:
                fig.add_trace(go.Scatter(x=fixed_x, y=fixed_y, mode="markers", marker=dict(size=8, symbol="square", color="#333333"), name="Fixed Support"))
            if loaded_x:
                fig.add_trace(go.Scatter(x=loaded_x, y=loaded_y, mode="markers", marker=dict(size=9, symbol="circle", color="#ef4444"), name="Load"))
            for xp, yp, ax, ay, txt in loaded_arrows:
                fig.add_annotation(x=xp, y=yp, ax=ax, ay=ay, xref="x", yref="y", axref="pixel", ayref="pixel", text=txt, showarrow=True, arrowhead=2, arrowcolor="#ef4444", font=dict(color="#ef4444", size=14, family="Courier Prime"))

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1, gridcolor="#e2e8f0"),
                yaxis=dict(title="Y (mm)", gridcolor="#e2e8f0"),
                height=450,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            return fig

        elif view_style == "stress_masked":
            fig = go.Figure()
            if smooth:
                # Interpolated contour without lines
                fig.add_trace(go.Contour(
                    z=s_II_masked,
                    x=x_coords,
                    y=y_coords,
                    colorscale=colormap,
                    zmin=min_val,
                    zmax=max_val,
                    contours_coloring="heatmap",
                    line_width=0,
                    colorbar=dict(
                        title=dict(text="Principal Stress [MPa]", font=dict(color="#f8fafc", size=11)),
                        tickfont=dict(color="#94a3b8", size=10),
                        len=0.85
                    ),
                    connectgaps=False,
                    name="Stress"
                ))
            else:
                # Blocky cubes
                fig.add_trace(go.Heatmap(
                    z=s_II_masked,
                    x=x_coords,
                    y=y_coords,
                    colorscale=colormap,
                    zmin=min_val,
                    zmax=max_val,
                    colorbar=dict(
                        title=dict(text="Principal Stress [MPa]", font=dict(color="#f8fafc", size=11)),
                        tickfont=dict(color="#94a3b8", size=10),
                        len=0.85
                    ),
                    hoverongaps=False,
                    name="Stress"
                ))
            if fixed_x:
                fig.add_trace(go.Scatter(x=fixed_x, y=fixed_y, mode="markers", marker=dict(size=8, symbol="square", color="#333333"), name="Fixed Support"))
            if loaded_x:
                fig.add_trace(go.Scatter(x=loaded_x, y=loaded_y, mode="markers", marker=dict(size=9, symbol="circle", color="#ef4444"), name="Load"))
            for xp, yp, ax, ay, txt in loaded_arrows:
                fig.add_annotation(x=xp, y=yp, ax=ax, ay=ay, xref="x", yref="y", axref="pixel", ayref="pixel", text=txt, showarrow=True, arrowhead=2, arrowcolor="#ffffff", font=dict(color="#ffffff", size=14, family="Courier Prime"))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#020617",
                xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1, gridcolor="#334155"),
                yaxis=dict(title="Y (mm)", gridcolor="#334155"),
                height=450,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            return fig

        return self.to_plotly_heatmap(res, threshold=threshold, colormap=colormap)

    def to_3d_params(
        self,
        res: SIMPResult,
        threshold: float = 0.40,
        depth_z_mm: float = 10.0,
        voxel_size_mm: float = 0.6
    ):
        """
        Converts 2D continuous density field into 3D volumetric beams for PicoGK solid synthesis.
        Extracts solid voxel centers, depth extrusions, and orthogonal connectors.
        """
        from PicoGK_Dashboard.core.models import BeamParam, GroundStructureTrussParams

        rho = res.density_matrix
        nely, nelx = rho.shape
        dx, dy = self.dx, self.dy
        r_base = 0.55 * max(dx, dy)
        beams = []

        for ely in range(nely):
            for elx in range(nelx):
                val = rho[ely, elx]
                if val >= threshold:
                    xc = (elx + 0.5) * dx
                    yc = (ely + 0.5) * dy
                    r = r_base * math.sqrt(val)

                    # Extrusion beam in Z
                    beams.append(BeamParam(
                        x1=xc, y1=yc, z1=-depth_z_mm / 2.0,
                        x2=xc, y2=yc, z2=+depth_z_mm / 2.0,
                        radius=r, force=val
                    ))

                    # Horizontal neighbor connection
                    if elx + 1 < nelx and rho[ely, elx + 1] >= threshold:
                        beams.append(BeamParam(
                            x1=xc, y1=yc, z1=0.0,
                            x2=(elx + 1.5) * dx, y2=yc, z2=0.0,
                            radius=r, force=val
                        ))

                    # Vertical neighbor connection
                    if ely + 1 < nely and rho[ely + 1, elx] >= threshold:
                        beams.append(BeamParam(
                            x1=xc, y1=yc, z1=0.0,
                            x2=xc, y2=(ely + 1.5) * dy, z2=0.0,
                            radius=r, force=val
                        ))

        return GroundStructureTrussParams(
            voxel_size_mm=voxel_size_mm,
            depth_z_mm=depth_z_mm,
            beams=beams
        )

