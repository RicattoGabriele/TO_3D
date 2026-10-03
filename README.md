# CE-3D: 3D Continuum Topology Optimization with Linearized Buckling Stability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-scipy%2Fnumpy-green.svg)](https://scipy.org)
[![Tests: 10/10 Passed](https://img.shields.io/badge/tests-100%25%20passed-brightgreen.svg)]()

A high-performance, CPU-efficient Python framework for **3D Multi-Objective Continuum Topology Optimization**, balancing global elastic compliance minimization and linearized buckling stability. Developed as an extensible computational engineering foundation inspired by generative design systems such as Leap71's *Noyron*.

---

## 🔬 Theoretical Foundations & Mathematical Log

A central pillar of this project is the rigorous academic and mathematical documentation accompanying all code implementations.

- **Full Mathematical Specification:** [`docs/TO_3D_log.md`](docs/TO_3D_log.md)
- **Academic Paper (PDF):** [`docs/TO_3D_log.pdf`](docs/TO_3D_log.pdf)
- **Reference 2D Document:** [`docs/TO_2D_reference.pdf`](docs/TO_2D_reference.pdf)

### Key Formulations:
1. **8-Node Hexahedral Elements ($H8$):** $24 \times 24$ element stiffness matrix $\mathbf{k}_0^e$ using $2 \times 2 \times 2$ Gauss quadrature, verified to possess exactly 6 rigid body zero-modes and 18 positive elastic strain modes.
2. **Linearized Buckling Stability (LBA):** Generalized eigenvalue formulation $(\mathbf{K} + \lambda_i \mathbf{G}) \boldsymbol{\phi}_i = \mathbf{0}$, parameterized via reciprocal eigenvalue $\mu_i = 1/\lambda_i$.
3. **6-Basis Geometric Stiffness Decomposition:** The 3D geometric stiffness matrix $\mathbf{G}_e$ is decomposed into 6 constant, precomputed basis matrices:
   $$\mathbf{G}_e = \sum_{k=1}^6 \sigma_{e,k} \mathbf{G}_{0,k}$$
   eliminating the need for repeated numerical quadrature inside the optimization loop.
4. **Exact Adjoint Sensitivity Analysis:** Solves the adjoint displacement state with a single back-solve using the pre-factored stiffness matrix, matching finite differences to within $\text{RelErr} < 5 \times 10^{-9}$.
5. **Adaptive Dynamic Sensitivity Scaling:** Balances disparate gradient magnitudes between compliance and buckling:
   $$\frac{\partial F^{(i)}}{\partial x_e} = (1 - \alpha) \frac{1}{S_C^{(i)}} \frac{\partial C^{(i)}}{\partial x_e} + \alpha \frac{1}{S_\mu^{(i)}} \frac{\partial \mu_1^{(i)}}{\partial x_e}$$
6. **Optimality Criteria (OC) Bisection:** Unconditionally stable density update with active bounds and move limits ($\zeta = 0.2$).

---

## ⚡ Performance & Low-Memory Architecture (< 4GB RAM)

Designed specifically to execute smoothly on standard mid-range desktop or laptop hardware (e.g. AMD Ryzen 7 4700U with 8GB RAM) without requiring dedicated GPU accelerators:
- **Solver Options:** Preconditioned Conjugate Gradient (**PCG**) with Jacobi diagonal preconditioning and displacement warm-starting (yielding up to 18x memory reduction over direct solvers) alongside SuperLU direct solver fallback.
- **Sparsity & Vectorization:** Precomputed indexing vectors (`iK`, `jK`) and 3D convolution spatial filtering via `scipy.ndimage.convolve`.
- **Memory Footprint:** $\approx 12\text{ KB}$ per element ($\approx 12\text{ MB}$ for $10 \times 10 \times 10$, $< 650\text{ MB}$ for 54,000 elements).

---

## 📁 Repository Structure

```text
CE-3D/
├── docs/
│   ├── TO_3D_log.md             # Complete academic mathematical log (Markdown)
│   ├── TO_3D_log.pdf            # Compiled academic paper (PDF)
│   └── TO_2D_reference.pdf      # Original 2D paper baseline
├── physics_engine/
│   ├── __init__.py              # Package entry point
│   ├── simp_engine_3d.py        # Core 3D multi-objective TO solver
│   ├── simp_engine_2d.py        # 2D plane stress TO reference solver
│   └── ground_structure.py      # Discrete truss ground structure optimizer
├── dashboard/
│   └── app.py                   # Interactive Streamlit dashboard with Plotly 3D visualizer
├── examples/
│   └── run_3d_cantilever.py     # Standalone CLI example script
├── tests/
│   └── test_simp_engine_3d.py   # Unit & benchmark test suite (100% pass)
├── outputs/
│   └── .gitkeep                 # Generated STL meshes and visualization outputs
├── .gitignore                   # Standard clean gitignore
├── requirements.txt             # Pip dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/username/CE-3D.git
cd CE-3D
pip install -r requirements.txt
```

### 2. Run the Standalone 3D Example
Run an end-to-end 3D cantilever optimization with buckling stability and automated STL export:
```bash
python examples/run_3d_cantilever.py
```
The resulting watertight `.stl` file is saved directly into `outputs/cantilever_3d_optimized.stl`.

### 3. Launch the Interactive Dashboard
Launch the interactive Streamlit GUI with real-time Plotly 3D isosurface rendering:
```bash
streamlit run dashboard/app.py
```

### 4. Run the Test Suite
Verify that all 10 unit and spectral tests pass:
```bash
pytest tests/test_simp_engine_3d.py -v
```

---

## 🛠️ STL & Generative Design Pipeline
The output density tensor $\mathbf{x} \in [0, 1]^{nely \times nelx \times nelz}$ can be converted directly into a watertight, manifold triangular mesh via:
- **Trimesh Box Voxel Representation:** Built directly into `opt.export_stl(res, "output.stl", threshold=0.35)`.
- **PicoGK / OpenVDB Smoothing:** Compatible with implicit signed distance field (SDF) smoothing for additive manufacturing and CNC machining.

---

## 📚 References
1. Ferrari, F., & Sigmund, O. (2020). Revisiting topology optimization with buckling constraints. *Structural and Multidisciplinary Optimization*, 61(4), 1401-1415.
2. Sigmund, O. (2001). A 99 line topology optimization code written in Matlab. *Structural and Multidisciplinary Optimization*, 21(2), 120-127.
3. Marler, R. T., & Arora, J. S. (2004). Survey of multi-objective optimization methods for engineering. *Structural and Multidisciplinary Optimization*, 26(6), 369-395.
4. Sigmund, O. (1997). On the design of compliant mechanisms using topology optimization. *Mechanics of Structures and Machines*, 25(4), 493-524.
5. Neves, M. M., Rodrigues, H., & Guedes, J. M. (1995). Generalized topology design of structures with a buckling load criterion. *Structural Optimization*, 10(2), 71-78.
6. Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications*. Springer Science & Business Media.
7. Ferrari, F., Sigmund, O., & Guest, J. K. (2021). Topology optimization with linearized buckling criteria in 250 lines of Matlab. *Structural and Multidisciplinary Optimization*, 63, 3045–3066.
8. Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and robust formulations in topology optimization. *Structural and Multidisciplinary Optimization*, 43(6), 767–784.
