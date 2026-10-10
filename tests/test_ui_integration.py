"""
Automated Streamlit Headless Integration Test Suite (test_ui_integration.py).

Validates:
1. Headless dashboard execution: dashboard/app.py starts and renders with 0 exceptions via AppTest.
2. Solver selector presence: dropdown options include 'AMG', 'PCG + Jacobi', 'Direct'.
3. Optimizer selector presence: dropdown options include 'MMA', 'OC'.
4. Optimization Mode selector presence: dropdown options include 'Compliance', 'Buckling Max',
   'Thermo-Elastic Robustness', 'Heat Exchanger / Thermal Compliance'.
5. Thermal boundary syntax parsing: SymPy evaluation of coordinates ('Lx', 'Ly', 'Lz') for
   'TEMP FACE', 'TEMP NODE', 'TEMP BOX', 'HEAT' directives, and application to SIMPOptimizer3D.
6. 3D Temperature field visualization: build_domain_preview_figure trace generation with
   view_mode='temperature', Inferno colormap, and temperature colorbar title.
"""

import os
import sys
import unittest
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_PATH = os.path.join(PROJECT_ROOT, "dashboard", "app.py")

from physics_engine.simp_engine_3d import SIMPOptimizer3D, SIMPResult3D
from dashboard.app import build_domain_preview_figure


class TestUIIntegration(unittest.TestCase):
    """Integration test suite for Streamlit Dashboard multi-physics controls."""

    def test_headless_dashboard_render_clean(self):
        """
        Test 1: Verify dashboard/app.py renders cleanly without exceptions using AppTest.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        self.assertEqual(
            len(at.exception), 0,
            f"Dashboard must start with 0 exceptions, but encountered: {[e.value for e in at.exception]}"
        )

    def test_solver_selector_options_present(self):
        """
        Test 2: Verify presence of Solver dropdown with options 'AMG', 'PCG + Jacobi', 'Direct'.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        # Find selectbox for solver (labeled 'Solver' or 'Linear Solver')
        solver_sbs = [sb for sb in at.selectbox if "solver" in sb.label.lower()]
        self.assertGreater(
            len(solver_sbs), 0,
            f"Expected a Solver selectbox in sidebar. Found selectboxes: {[sb.label for sb in at.selectbox]}"
        )
        solver_sb = solver_sbs[0]
        options = solver_sb.options

        # Options must contain AMG, PCG, and Direct
        has_amg = any("AMG" in opt for opt in options)
        has_pcg = any("PCG" in opt for opt in options)
        has_direct = any("Direct" in opt for opt in options)

        self.assertTrue(has_amg, f"Solver options {options} must include 'AMG'.")
        self.assertTrue(has_pcg, f"Solver options {options} must include 'PCG + Jacobi'.")
        self.assertTrue(has_direct, f"Solver options {options} must include 'Direct'.")

    def test_optimizer_selector_options_present(self):
        """
        Test 3: Verify presence of Optimizer dropdown with options 'MMA', 'OC'.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        opt_sbs = [sb for sb in at.selectbox if "optimizer" in sb.label.lower()]
        self.assertGreater(
            len(opt_sbs), 0,
            f"Expected an Optimizer selectbox in sidebar. Found selectboxes: {[sb.label for sb in at.selectbox]}"
        )
        opt_sb = opt_sbs[0]
        options = [o.upper() for o in opt_sb.options]

        self.assertIn("MMA", options, f"Optimizer options {opt_sb.options} must include 'MMA'.")
        self.assertIn("OC", options, f"Optimizer options {opt_sb.options} must include 'OC'.")

    def test_optimization_mode_selector_options_present(self):
        """
        Test 4: Verify presence of Mode dropdown with options:
        'Compliance', 'Buckling Max', 'Thermo-Elastic Robustness', 'Heat Exchanger / Thermal Compliance'.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        mode_sbs = [sb for sb in at.selectbox if "mode" in sb.label.lower()]
        self.assertGreater(
            len(mode_sbs), 0,
            f"Expected an Optimization Mode selectbox in sidebar. Found selectboxes: {[sb.label for sb in at.selectbox]}"
        )
        mode_sb = mode_sbs[0]
        options = mode_sb.options

        has_compliance = any("compliance" in o.lower() for o in options)
        has_buckling = any("buckling" in o.lower() for o in options)
        has_thermo = any("thermo-elastic" in o.lower() or "thermo_elastic" in o.lower() for o in options)
        has_heat_ex = any("heat exchanger" in o.lower() or "thermal compliance" in o.lower() for o in options)

        self.assertTrue(has_compliance, f"Mode options {options} must include Compliance.")
        self.assertTrue(has_buckling, f"Mode options {options} must include Buckling Max.")
        self.assertTrue(has_thermo, f"Mode options {options} must include Thermo-Elastic Robustness.")
        self.assertTrue(has_heat_ex, f"Mode options {options} must include Heat Exchanger / Thermal Compliance.")

    def test_sympy_thermal_boundary_syntax_parsing(self):
        """
        Test 5: Verify sympy.sympify parsing of Section 5 Thermal BC directives:
        - TEMP FACE <left|right|bottom|top|front|back> <T>
        - TEMP NODE X Y Z <T>
        - TEMP BOX X1 X2 Y1 Y2 Z1 Z2 <T>
        - HEAT X Y Z Q
        - HEAT BOX X1 X2 Y1 Y2 Z1 Z2 Q
        with symbolic coordinate expressions involving Lx, Ly, Lz (e.g., Lx/2, Ly/2, Lz/2).
        Verifies correct mapping into SIMPOptimizer3D boundary condition state.
        """
        import sympy

        nelx, nely, nelz = 10, 6, 4
        dx, dy, dz = 2.0, 2.0, 2.0
        Lx = float(nelx * dx)
        Ly = float(nely * dy)
        Lz = float(nelz * dz)
        safe_dict = {"Lx": Lx, "Ly": Ly, "Lz": Lz, "dx": dx, "dy": dy, "dz": dz}

        opt = SIMPOptimizer3D(nelx=nelx, nely=nely, nelz=nelz, dx=dx, dy=dy, dz=dz)

        thermal_input = """
        # Section 5 Thermal Directives
        TEMP FACE left 0.0
        TEMP FACE right 100.0
        TEMP NODE Lx/2 Ly/2 Lz/2 50.0
        TEMP BOX 0 Lx/4 0 Ly/4 0 Lz/4 25.0
        HEAT Lx/2 Ly/2 Lz/2 40.0
        HEAT BOX Lx/4 3*Lx/4 Ly/4 3*Ly/4 0 Lz/2 80.0
        """

        parsed_faces = []
        parsed_nodes = []
        parsed_boxes = []
        parsed_heats = []
        parsed_heat_boxes = []

        for line in thermal_input.strip().split('\n'):
            line = line.split('#')[0].strip().upper()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "TEMP" and len(parts) >= 3:
                sub = parts[1]
                if sub == "FACE" and len(parts) >= 4:
                    face_name = parts[2].lower()
                    t_val = float(sympy.sympify(parts[3].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    parsed_faces.append((face_name, t_val))
                    opt.fix_thermal_face(face_name, temp=t_val)
                elif sub == "NODE" and len(parts) >= 6:
                    x = float(sympy.sympify(parts[2].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    y = float(sympy.sympify(parts[3].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    z = float(sympy.sympify(parts[4].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    t_val = float(sympy.sympify(parts[5].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    parsed_nodes.append((x, y, z, t_val))
                    n_i = int(np.clip(round(x / dx), 0, nelx))
                    n_j = int(np.clip(round(y / dy), 0, nely))
                    n_k = int(np.clip(round(z / dz), 0, nelz))
                    opt.fix_thermal_node(n_i, n_j, n_k, temp=t_val)
                elif sub == "BOX" and len(parts) >= 9:
                    x1 = float(sympy.sympify(parts[2].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    x2 = float(sympy.sympify(parts[3].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    y1 = float(sympy.sympify(parts[4].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    y2 = float(sympy.sympify(parts[5].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    z1 = float(sympy.sympify(parts[6].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    z2 = float(sympy.sympify(parts[7].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    t_val = float(sympy.sympify(parts[8].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    parsed_boxes.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2), min(z1, z2), max(z1, z2), t_val))

            elif cmd == "HEAT" and len(parts) >= 5:
                if parts[1] == "BOX" and len(parts) >= 9:
                    x1 = float(sympy.sympify(parts[2].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    x2 = float(sympy.sympify(parts[3].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    y1 = float(sympy.sympify(parts[4].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    y2 = float(sympy.sympify(parts[5].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    z1 = float(sympy.sympify(parts[6].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    z2 = float(sympy.sympify(parts[7].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    q_val = float(sympy.sympify(parts[8].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    parsed_heat_boxes.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2), min(z1, z2), max(z1, z2), q_val))
                else:
                    offset = 1 if parts[1] == "NODE" else 0
                    x = float(sympy.sympify(parts[1 + offset].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    y = float(sympy.sympify(parts[2 + offset].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    z = float(sympy.sympify(parts[3 + offset].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    q_val = float(sympy.sympify(parts[4 + offset].replace("LX", "Lx").replace("LY", "Ly").replace("LZ", "Lz"), locals=safe_dict))
                    parsed_heats.append((x, y, z, q_val))
                    n_i = int(np.clip(round(x / dx), 0, nelx))
                    n_j = int(np.clip(round(y / dy), 0, nely))
                    n_k = int(np.clip(round(z / dz), 0, nelz))
                    opt.add_heat_source(n_i, n_j, n_k, q=q_val)

        # Verify parsing counts
        self.assertEqual(len(parsed_faces), 2, "Expected 2 parsed thermal faces.")
        self.assertEqual(len(parsed_nodes), 1, "Expected 1 parsed thermal node.")
        self.assertEqual(len(parsed_boxes), 1, "Expected 1 parsed thermal box.")
        self.assertEqual(len(parsed_heats), 1, "Expected 1 parsed point heat source.")
        self.assertEqual(len(parsed_heat_boxes), 1, "Expected 1 parsed heat box.")

        # Verify node evaluation: Lx/2, Ly/2, Lz/2 evaluates to 10.0, 6.0, 4.0
        self.assertEqual(parsed_nodes[0], (10.0, 6.0, 4.0, 50.0))
        # Verify heat source applied at node (5, 3, 2)
        target_nid = opt.node_id(5, 3, 2)
        self.assertEqual(opt.heat_source_vector[target_nid], 40.0)
        # Verify left and right face nodes are fixed in optimizer
        left_corner_nid = opt.node_id(0, 0, 0)
        right_corner_nid = opt.node_id(nelx, nely, nelz)
        self.assertIn(left_corner_nid, opt.fixed_thermal_nodes)
        self.assertEqual(opt.fixed_thermal_nodes[left_corner_nid], 0.0)
        self.assertIn(right_corner_nid, opt.fixed_thermal_nodes)
        self.assertEqual(opt.fixed_thermal_nodes[right_corner_nid], 100.0)

    def test_3d_temperature_field_view_mode_and_inferno_colormap(self):
        """
        Test 6: Verify build_domain_preview_figure generates a 3D Mesh3d trace with
        Inferno colormap and temperature colorbar title when view_mode='temperature'.
        """
        nelx, nely, nelz = 4, 4, 4
        dx, dy, dz = 1.0, 1.0, 1.0

        # Construct a synthetic SIMPResult3D with temperature_field and density
        density = np.ones((nelz, nely, nelx), dtype=np.float64)
        t_field = np.linspace(20.0, 120.0, nelx * nely * nelz).reshape((nelz, nely, nelx))

        res = SIMPResult3D(
            success=True,
            density_matrix=density,
            compliance=1.0,
            volume_fraction=1.0,
            iterations_run=5,
            execution_time_sec=0.1,
            nelx=nelx, nely=nely, nelz=nelz,
            dx=dx, dy=dy, dz=dz,
            temperatures=t_field.ravel()
        )
        # Attach temperature_field attribute
        res.temperature_field = t_field

        fig = build_domain_preview_figure(
            nelx=nelx, nely=nely, nelz=nelz,
            dx=dx, dy=dy, dz=dz,
            fixed_nodes_coords=[],
            applied_loads=[],
            res=res,
            threshold=0.35,
            view_mode="temperature"
        )

        # Inspect traces in Plotly figure
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        self.assertGreater(
            len(mesh_traces), 0,
            "build_domain_preview_figure must produce a Mesh3d trace when view_mode='temperature'."
        )

        temp_trace = mesh_traces[0]
        # Verify Inferno colormap (case-insensitive check)
        colorscale = str(temp_trace.colorscale).lower()
        self.assertIn("inferno", colorscale, f"Mesh3d colorscale must be 'Inferno', got: {temp_trace.colorscale}")

        # Verify colorbar title relates to Temperature (°C / K)
        cbar_title = ""
        if hasattr(temp_trace, "colorbar") and temp_trace.colorbar is not None:
            cbar_dict = temp_trace.colorbar.to_plotly_json() if hasattr(temp_trace.colorbar, "to_plotly_json") else dict(temp_trace.colorbar)
            title_info = cbar_dict.get("title", "")
            cbar_title = str(title_info.get("text", title_info) if isinstance(title_info, dict) else title_info)

        has_t_label = ("t" in cbar_title.lower() or "temp" in cbar_title.lower() or "k" in cbar_title or "°c" in cbar_title.lower())
        self.assertTrue(has_t_label, f"Colorbar title must indicate Temperature (T, K, or °C), got: '{cbar_title}'")


if __name__ == "__main__":
    unittest.main()
