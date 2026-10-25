# 3D Topology optimization with LBA
Still under development. For theory and formulation check TO.
## Repository Structure

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

## Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/username/CE-3D.git
cd CE-3D
pip install -r requirements.txt
```

### 2. Run the Example
Run an end-to-end 3D cantilever optimization with buckling stability and automated STL export:
```bash
python examples/run_3d_cantilever.py
```
The resulting watertight `.stl` file is saved directly into `outputs/cantilever_3d_optimized.stl`.

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Tests
```bash
pytest tests/test_simp_engine_3d.py -v
```

