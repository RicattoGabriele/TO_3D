"""
PicoGK Aerospace Component Generator - Template Engine & Code Synthesizer.
Provides parameter serialization to JSON (matching C# contract) and synthesis of
reproducible, standalone C# source code for aerospace engineers to inspect and execute.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel

from PicoGK_Dashboard.core.models import (
    ComponentParams,
    ComponentType,
    ConduitFlangeParams,
    GroundStructureTrussParams,
    ImportedMeshParams,
    LatticeBlockParams,
)


def write_params_json(params: BaseModel, output_path: str) -> str:
    """
    Serializes a validated parameter model into JSON format matching the C# runner contract.
    Ensures parent directories exist and returns absolute file path.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Use model_dump_json or model_dump
    data = params.model_dump()
    if isinstance(data.get("component_type"), ComponentType):
        data["component_type"] = data["component_type"].value

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(out_p)


def generate_csharp_source(
    params: Union[LatticeBlockParams, ConduitFlangeParams, ImportedMeshParams, BaseModel],
    output_path: Optional[str] = None
) -> str:
    """
    Synthesizes a standalone, beautifully formatted C# source file implementing
    the exact PicoGK computational geometry pipeline for the given component.
    """
    if isinstance(params, LatticeBlockParams) or (
        hasattr(params, "component_type") and params.component_type == ComponentType.LATTICE_BLOCK
    ):
        code = _generate_lattice_csharp(params)
    elif isinstance(params, ConduitFlangeParams) or (
        hasattr(params, "component_type") and params.component_type == ComponentType.CONDUIT_FLANGE
    ):
        code = _generate_conduit_csharp(params)
    elif isinstance(params, ImportedMeshParams) or (
        hasattr(params, "component_type") and params.component_type == ComponentType.IMPORTED_MESH
    ):
        code = _generate_imported_mesh_csharp(params)
    elif isinstance(params, GroundStructureTrussParams) or (
        hasattr(params, "component_type") and params.component_type == ComponentType.GROUND_STRUCTURE
    ):
        code = _generate_ground_structure_csharp(params)
    else:
        raise ValueError(f"Unsupported component parameter type: {type(params)}")

    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(code)

    return code


def _generate_lattice_csharp(params: LatticeBlockParams) -> str:
    """Generates standalone C# source code for a Parametric Lattice Block in PicoGK."""
    plates_comment = "// Top and bottom solid mounting plates included" if params.include_solid_plates else "// Open lattice structure (no solid skin plates)"
    solid_plates_block = ""
    if params.include_solid_plates and params.skin_thickness_mm > 0:
        solid_plates_block = f"""
            // Add top and bottom mounting plates / solid skins
            if (bIncludeSolidPlates && fSkinThickness > 0f)
            {{
                var plateLattice = new Lattice(lib);
                float step = MathF.Max(fSkinThickness * 0.5f, 0.5f);
                for (float x = 0f; x <= fSizeX; x += step)
                {{
                    // Bottom mounting skin
                    plateLattice.AddBeam(
                        new Vector3(x, 0, fSkinThickness * 0.5f),
                        fSkinThickness * 0.5f,
                        new Vector3(x, fSizeY, fSkinThickness * 0.5f),
                        fSkinThickness * 0.5f,
                        true
                    );
                    // Top mounting skin
                    plateLattice.AddBeam(
                        new Vector3(x, 0, fSizeZ - fSkinThickness * 0.5f),
                        fSkinThickness * 0.5f,
                        new Vector3(x, fSizeY, fSizeZ - fSkinThickness * 0.5f),
                        fSkinThickness * 0.5f,
                        true
                    );
                }}

                var voxPlates = new Voxels(plateLattice);
                voxPlates.Trim(bounds);
                voxLattice.BoolAdd(voxPlates);
            }}
"""

    return f"""// ============================================================================
// PicoGK Computational Engineering - Standalone Generated Component
// Component: Parametric Aerospace Lattice Block ({params.lattice_type})
// Generated: Auto-synthesized from Natural Language Engineering Specifications
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using PicoGK;

namespace PicoGK_Generated
{{
    public static class Program
    {{
        public static void Main(string[] args)
        {{
            float fVoxelSizeMM = {params.voxel_size_mm:.4f}f;
            string outputStl = args.Length > 0 ? args[0] : "LatticeBlock_Output.stl";
            string outputObj = args.Length > 1 ? args[1] : "LatticeBlock_Output.obj";

            Console.WriteLine("Initializing PicoGK OpenVDB Geometry Engine...");
            using var lib = new Library(fVoxelSizeMM);

            try
            {{
                Console.WriteLine("Generating Lattice Block with Parameters:");
                Console.WriteLine(" - Dimensions: {params.size_x_mm:.1f} x {params.size_y_mm:.1f} x {params.size_z_mm:.1f} mm");
                Console.WriteLine(" - Lattice Type: {params.lattice_type}");
                Console.WriteLine(" - Cell Size: {params.cell_size_mm:.2f} mm");
                Console.WriteLine(" - Strut Radius: {params.strut_radius_mm:.2f} mm");
                Console.WriteLine(" - Node Radius: {params.node_radius_mm:.2f} mm");
                Console.WriteLine(" - Voxel Resolution: {params.voxel_size_mm:.3f} mm");
                {plates_comment}

                Mesh mesh = BuildLatticeBlock(
                    lib: lib,
                    fVoxelSizeMM: fVoxelSizeMM,
                    fSizeX: {params.size_x_mm:.3f}f,
                    fSizeY: {params.size_y_mm:.3f}f,
                    fSizeZ: {params.size_z_mm:.3f}f,
                    fCellSize: {params.cell_size_mm:.3f}f,
                    fStrutRadius: {params.strut_radius_mm:.3f}f,
                    fNodeRadius: {params.node_radius_mm:.3f}f,
                    fSkinThickness: {params.skin_thickness_mm:.3f}f,
                    bIncludeSolidPlates: {str(params.include_solid_plates).lower()}
                );

                Console.WriteLine($"Mesh generated successfully: {{mesh.nVertexCount()}} vertices, {{mesh.nTriangleCount()}} triangles.");

                // Export slicer-ready 3D mesh files
                SaveToStl(mesh, outputStl);
                SaveToObj(mesh, outputObj);
                Console.WriteLine($"Exported: {{outputStl}} & {{outputObj}}");
            }}
            catch (Exception ex)
            {{
                Console.Error.WriteLine($"[PICOGK_ERROR] {{ex.GetType().Name}}: {{ex.Message}}\\n{{ex.StackTrace}}");
                Environment.Exit(1);
            }}
        }}

        public static Mesh BuildLatticeBlock(
            Library lib,
            float fVoxelSizeMM,
            float fSizeX,
            float fSizeY,
            float fSizeZ,
            float fCellSize,
            float fStrutRadius,
            float fNodeRadius,
            float fSkinThickness,
            bool bIncludeSolidPlates)
        {{
            var lattice = new Lattice(lib);

            int nCellsX = (int)MathF.Ceiling(fSizeX / fCellSize);
            int nCellsY = (int)MathF.Ceiling(fSizeY / fCellSize);
            int nCellsZ = (int)MathF.Ceiling(fSizeZ / fCellSize);

            // Populate unit cells ({params.lattice_type} Topology)
            for (int x = 0; x <= nCellsX; x++)
            {{
                for (int y = 0; y <= nCellsY; y++)
                {{
                    for (int z = 0; z <= nCellsZ; z++)
                    {{
                        Vector3 corner = new Vector3(x * fCellSize, y * fCellSize, z * fCellSize);
                        lattice.AddSphere(corner, fNodeRadius);

                        if (x < nCellsX)
                            lattice.AddBeam(corner, fStrutRadius, new Vector3((x + 1) * fCellSize, y * fCellSize, z * fCellSize), fStrutRadius, true);
                        if (y < nCellsY)
                            lattice.AddBeam(corner, fStrutRadius, new Vector3(x * fCellSize, (y + 1) * fCellSize, z * fCellSize), fStrutRadius, true);
                        if (z < nCellsZ)
                            lattice.AddBeam(corner, fStrutRadius, new Vector3(x * fCellSize, y * fCellSize, (z + 1) * fCellSize), fStrutRadius, true);

                        if (x < nCellsX && y < nCellsY && z < nCellsZ)
                        {{
                            Vector3 center = new Vector3(
                                (x + 0.5f) * fCellSize,
                                (y + 0.5f) * fCellSize,
                                (z + 0.5f) * fCellSize
                            );
                            lattice.AddSphere(center, fNodeRadius);

                            lattice.AddBeam(center, fStrutRadius, corner, fStrutRadius, true);
                            lattice.AddBeam(center, fStrutRadius, new Vector3((x + 1) * fCellSize, y * fCellSize, z * fCellSize), fStrutRadius, true);
                            lattice.AddBeam(center, fStrutRadius, new Vector3(x * fCellSize, (y + 1) * fCellSize, z * fCellSize), fStrutRadius, true);
                            lattice.AddBeam(center, fStrutRadius, new Vector3((x + 1) * fCellSize, (y + 1) * fCellSize, z * fCellSize), fStrutRadius, true);

                            lattice.AddBeam(center, fStrutRadius, new Vector3(x * fCellSize, y * fCellSize, (z + 1) * fCellSize), fStrutRadius, true);
                            lattice.AddBeam(center, fStrutRadius, new Vector3((x + 1) * fCellSize, y * fCellSize, (z + 1) * fCellSize), fStrutRadius, true);
                            lattice.AddBeam(center, fStrutRadius, new Vector3(x * fCellSize, (y + 1) * fCellSize, (z + 1) * fCellSize), fStrutRadius, true);
                            lattice.AddBeam(center, fStrutRadius, new Vector3((x + 1) * fCellSize, (y + 1) * fCellSize, (z + 1) * fCellSize), fStrutRadius, true);
                        }}
                    }}
                }}
            }}

            // Convert lattice geometry into OpenVDB voxel field
            var voxLattice = new Voxels(lattice);

            // Trim cleanly to exact requested bounding dimensions
            BBox3 bounds = new BBox3(0, 0, 0, fSizeX, fSizeY, fSizeZ);
            voxLattice.Trim(bounds);
{solid_plates_block}
            // Extract watertight iso-surface triangle mesh via Marching Cubes
            return new Mesh(voxLattice);
        }}

        public static void SaveToStl(Mesh mesh, string filePath)
        {{
            if (mesh == null)
                throw new ArgumentNullException(nameof(mesh), "Cannot export null mesh to STL.");

            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {{
                Directory.CreateDirectory(directory);
            }}

            mesh.SaveToStlFile(filePath);
        }}

        public static void SaveToObj(Mesh mesh, string filePath)
        {{
            if (mesh == null)
                throw new ArgumentNullException(nameof(mesh), "Cannot export null mesh to OBJ.");

            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {{
                Directory.CreateDirectory(directory);
            }}

            using var writer = new StreamWriter(filePath, false, System.Text.Encoding.UTF8);
            writer.WriteLine("# PicoGK Computational Engineering 3D Model Export");
            writer.WriteLine($"# Vertices: {{mesh.nVertexCount()}}, Triangles: {{mesh.nTriangleCount()}}");

            int vertexCount = mesh.nVertexCount();
            for (int i = 0; i < vertexCount; i++)
            {{
                Vector3 v = mesh.vecVertexAt(i);
                writer.WriteLine($"v {{v.X.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Y.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Z.ToString(\"F4\", CultureInfo.InvariantCulture)}}");
            }}

            int triangleCount = mesh.nTriangleCount();
            for (int i = 0; i < triangleCount; i++)
            {{
                Triangle tri = mesh.oTriangleAt(i);
                writer.WriteLine($"f {{tri.A + 1}} {{tri.B + 1}} {{tri.C + 1}}");
            }}
        }}
    }}
}}
"""


def _generate_conduit_csharp(params: ConduitFlangeParams) -> str:
    """Generates standalone C# source code for a Parametric Conduit & Flange in PicoGK."""
    bolt_block = ""
    if params.bolt_hole_count > 0:
        bolt_block = f"""
            // Drill bolt circle array ({params.bolt_hole_count} holes at PCD {params.bolt_circle_radius_mm * 2.0:.2f}mm)
            if (nBoltCount > 0)
            {{
                var latHoles = new Lattice(lib);
                for (int i = 0; i < nBoltCount; i++)
                {{
                    float angle = i * 2.0f * MathF.PI / nBoltCount;
                    float x = fBoltCircleRad * MathF.Cos(angle);
                    float y = fBoltCircleRad * MathF.Sin(angle);

                    Vector3 holeStart = new Vector3(x, y, -5f);
                    Vector3 holeEnd = new Vector3(x, y, fFlangeThick + 5f);
                    latHoles.AddBeam(holeStart, fBoltHoleRad, holeEnd, fBoltHoleRad, true);
                }}
                var voxHoles = new Voxels(latHoles);
                voxSolid.BoolSubtract(voxHoles);
            }}
"""

    return f"""// ============================================================================
// PicoGK Computational Engineering - Standalone Generated Component
// Component: Parametric Aerospace Propellant Conduit & Bolted Flange Interface
// Generated: Auto-synthesized from Natural Language Engineering Specifications
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using PicoGK;

namespace PicoGK_Generated
{{
    public static class Program
    {{
        public static void Main(string[] args)
        {{
            float fVoxelSizeMM = {params.voxel_size_mm:.4f}f;
            string outputStl = args.Length > 0 ? args[0] : "ConduitFlange_Output.stl";
            string outputObj = args.Length > 1 ? args[1] : "ConduitFlange_Output.obj";

            Console.WriteLine("Initializing PicoGK OpenVDB Geometry Engine...");
            using var lib = new Library(fVoxelSizeMM);

            try
            {{
                Console.WriteLine("Generating Conduit & Flange with Parameters:");
                Console.WriteLine(" - Conduit Length: {params.conduit_length_mm:.1f} mm");
                Console.WriteLine(" - Outer Radius: {params.outer_radius_mm:.2f} mm (OD: {params.outer_radius_mm * 2.0:.1f} mm)");
                Console.WriteLine(" - Inner Bore Radius: {params.inner_radius_mm:.2f} mm (ID: {params.inner_radius_mm * 2.0:.1f} mm)");
                Console.WriteLine(" - Flange Outer Radius: {params.flange_radius_mm:.2f} mm (OD: {params.flange_radius_mm * 2.0:.1f} mm)");
                Console.WriteLine(" - Flange Thickness: {params.flange_thickness_mm:.2f} mm");
                Console.WriteLine(" - Bolt Pattern: {params.bolt_hole_count} holes, PCD = {params.bolt_circle_radius_mm * 2.0:.1f} mm, Hole Rad = {params.bolt_hole_radius_mm:.2f} mm");
                Console.WriteLine(" - Voxel Resolution: {params.voxel_size_mm:.3f} mm");

                Mesh mesh = BuildConduitFlange(
                    lib: lib,
                    fVoxelSizeMM: fVoxelSizeMM,
                    fLength: {params.conduit_length_mm:.3f}f,
                    fOuterRad: {params.outer_radius_mm:.3f}f,
                    fInnerRad: {params.inner_radius_mm:.3f}f,
                    fFlangeRad: {params.flange_radius_mm:.3f}f,
                    fFlangeThick: {params.flange_thickness_mm:.3f}f,
                    nBoltCount: {params.bolt_hole_count},
                    fBoltCircleRad: {params.bolt_circle_radius_mm:.3f}f,
                    fBoltHoleRad: {params.bolt_hole_radius_mm:.3f}f
                );

                Console.WriteLine($"Mesh generated successfully: {{mesh.nVertexCount()}} vertices, {{mesh.nTriangleCount()}} triangles.");

                // Export slicer-ready 3D mesh files
                SaveToStl(mesh, outputStl);
                SaveToObj(mesh, outputObj);
                Console.WriteLine($"Exported: {{outputStl}} & {{outputObj}}");
            }}
            catch (Exception ex)
            {{
                Console.Error.WriteLine($"[PICOGK_ERROR] {{ex.GetType().Name}}: {{ex.Message}}\\n{{ex.StackTrace}}");
                Environment.Exit(1);
            }}
        }}

        public static Mesh BuildConduitFlange(
            Library lib,
            float fVoxelSizeMM,
            float fLength,
            float fOuterRad,
            float fInnerRad,
            float fFlangeRad,
            float fFlangeThick,
            int nBoltCount,
            float fBoltCircleRad,
            float fBoltHoleRad)
        {{
            // 1. Outer Pipe Cylinder & Flange Base
            var latOuter = new Lattice(lib);
            latOuter.AddBeam(
                new Vector3(0, 0, 0),
                fOuterRad,
                new Vector3(0, 0, fLength),
                fOuterRad,
                true
            );

            latOuter.AddBeam(
                new Vector3(0, 0, 0),
                fFlangeRad,
                new Vector3(0, 0, fFlangeThick),
                fFlangeRad,
                true
            );

            // Reinforcing transition cone/fillet between flange and conduit
            float transitionHeight = MathF.Min(10f, (fLength - fFlangeThick) * 0.5f);
            if (transitionHeight > 0.5f)
            {{
                latOuter.AddBeam(
                    new Vector3(0, 0, fFlangeThick),
                    (fOuterRad + fFlangeRad) * 0.5f,
                    new Vector3(0, 0, fFlangeThick + transitionHeight),
                    fOuterRad,
                    true
                );
            }}

            var voxSolid = new Voxels(latOuter);

            // 2. Central Fluid Hollow Bore Subtraction
            var latBore = new Lattice(lib);
            latBore.AddBeam(
                new Vector3(0, 0, -5f),
                fInnerRad,
                new Vector3(0, 0, fLength + 5f),
                fInnerRad,
                true
            );
            var voxBore = new Voxels(latBore);
            voxSolid.BoolSubtract(voxBore);
{bolt_block}
            // 3. Extract Watertight Polygon Mesh
            return new Mesh(voxSolid);
        }}

        public static void SaveToStl(Mesh mesh, string filePath)
        {{
            if (mesh == null)
                throw new ArgumentNullException(nameof(mesh), "Cannot export null mesh to STL.");

            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {{
                Directory.CreateDirectory(directory);
            }}

            mesh.SaveToStlFile(filePath);
        }}

        public static void SaveToObj(Mesh mesh, string filePath)
        {{
            if (mesh == null)
                throw new ArgumentNullException(nameof(mesh), "Cannot export null mesh to OBJ.");

            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {{
                Directory.CreateDirectory(directory);
            }}

            using var writer = new StreamWriter(filePath, false, System.Text.Encoding.UTF8);
            writer.WriteLine("# PicoGK Computational Engineering 3D Model Export");
            writer.WriteLine($"# Vertices: {{mesh.nVertexCount()}}, Triangles: {{mesh.nTriangleCount()}}");

            int vertexCount = mesh.nVertexCount();
            for (int i = 0; i < vertexCount; i++)
            {{
                Vector3 v = mesh.vecVertexAt(i);
                writer.WriteLine($"v {{v.X.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Y.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Z.ToString(\"F4\", CultureInfo.InvariantCulture)}}");
            }}

            int triangleCount = mesh.nTriangleCount();
            for (int i = 0; i < triangleCount; i++)
            {{
                Triangle tri = mesh.oTriangleAt(i);
                writer.WriteLine($"f {{tri.A + 1}} {{tri.B + 1}} {{tri.C + 1}}");
            }}
        }}
    }}
}}
"""


def _generate_imported_mesh_csharp(params: ImportedMeshParams) -> str:
    """Generates standalone C# source code for an Imported Custom Mesh in PicoGK."""
    escaped_path = params.mesh_file_path.replace("\\", "\\\\")
    offset_comment = f"// Offset thickness: {params.offset_thickness_mm:+.2f} mm" if abs(params.offset_thickness_mm) > 0.001 else "// Zero offset (identity)"
    infill_comment = f"// Infill enabled: {params.lattice_type} ({params.cell_size_mm:.1f}mm cell, {params.strut_radius_mm:.2f}mm strut)" if params.apply_lattice_infill else "// Infill disabled (solid processed volume)"

    return f"""// ============================================================================
// PicoGK Computational Engineering - Standalone Generated Component
// Component: Imported Custom Mesh ({Path(params.mesh_file_path).name})
// Generated: Auto-synthesized from Natural Language Engineering Specifications
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using PicoGK;

namespace PicoGK_Generated
{{
    public static class Program
    {{
        public static void Main(string[] args)
        {{
            float fVoxelSizeMM = {params.voxel_size_mm:.4f}f;
            string outputStl = args.Length > 0 ? args[0] : "ImportedMesh_Output.stl";
            string outputObj = args.Length > 1 ? args[1] : "ImportedMesh_Output.obj";

            Console.WriteLine("Initializing PicoGK OpenVDB Geometry Engine...");
            using var lib = new Library(fVoxelSizeMM);

            try
            {{
                Console.WriteLine("Processing Imported Mesh with Parameters:");
                Console.WriteLine(" - Source Mesh: {escaped_path}");
                Console.WriteLine(" - Offset Thickness: {params.offset_thickness_mm:.2f} mm");
                Console.WriteLine(" - Apply Lattice Infill: {str(params.apply_lattice_infill).lower()}");
                Console.WriteLine(" - Voxel Resolution: {params.voxel_size_mm:.3f} mm");
                {offset_comment}
                {infill_comment}

                Mesh mesh = BuildImportedMesh(
                    lib: lib,
                    fVoxelSizeMM: fVoxelSizeMM,
                    strSourceMeshPath: @\"{params.mesh_file_path}\",
                    fOffsetThicknessMM: {params.offset_thickness_mm:.3f}f,
                    bApplyLatticeInfill: {str(params.apply_lattice_infill).lower()},
                    fCellSizeMM: {params.cell_size_mm:.3f}f,
                    fStrutRadiusMM: {params.strut_radius_mm:.3f}f,
                    fShellThicknessMM: {params.shell_thickness_mm:.3f}f
                );

                Console.WriteLine($\"Mesh processed successfully: {{mesh.nVertexCount()}} vertices, {{mesh.nTriangleCount()}} triangles.\");

                // Export slicer-ready 3D mesh files
                SaveToStl(mesh, outputStl);
                SaveToObj(mesh, outputObj);
                Console.WriteLine($\"Exported: {{outputStl}} & {{outputObj}}\");
            }}
            catch (Exception ex)
            {{
                Console.Error.WriteLine($\"[PICOGK_ERROR] {{ex.GetType().Name}}: {{ex.Message}}\\n{{ex.StackTrace}}\");
                Environment.Exit(1);
            }}
        }}

        public static Mesh BuildImportedMesh(
            Library lib,
            float fVoxelSizeMM,
            string strSourceMeshPath,
            float fOffsetThicknessMM,
            bool bApplyLatticeInfill,
            float fCellSizeMM,
            float fStrutRadiusMM,
            float fShellThicknessMM)
        {{
            if (!File.Exists(strSourceMeshPath))
                throw new FileNotFoundException($\"Source mesh not found: {{strSourceMeshPath}}\");

            Mesh srcMesh = strSourceMeshPath.EndsWith(\".stl\", StringComparison.OrdinalIgnoreCase)
                ? Mesh.mshFromStlFile(strSourceMeshPath, Mesh.EStlUnit.AUTO, 1.0f, null, lib)
                : LoadWavefrontObj(lib, strSourceMeshPath);

            Voxels voxels = new Voxels(srcMesh);

            if (MathF.Abs(fOffsetThicknessMM) > 0.001f)
            {{
                voxels.Offset(fOffsetThicknessMM);
            }}

            if (bApplyLatticeInfill)
            {{
                BBox3 bounds = srcMesh.oBoundingBox();
                var lattice = new Lattice(lib);
                float cs = fCellSizeMM;
                float sr = fStrutRadiusMM;

                float minX = bounds.vecMin.X - cs;
                float maxX = bounds.vecMax.X + cs;
                float minY = bounds.vecMin.Y - cs;
                float maxY = bounds.vecMax.Y + cs;
                float minZ = bounds.vecMin.Z - cs;
                float maxZ = bounds.vecMax.Z + cs;

                for (float x = minX; x < maxX; x += cs)
                {{
                    for (float y = minY; y < maxY; y += cs)
                    {{
                        for (float z = minZ; z < maxZ; z += cs)
                        {{
                            Vector3 vMin = new Vector3(x, y, z);
                            Vector3 vCenter = vMin + new Vector3(cs * 0.5f, cs * 0.5f, cs * 0.5f);

                            Vector3[] corners = new Vector3[]
                            {{
                                vMin + new Vector3(0, 0, 0),
                                vMin + new Vector3(cs, 0, 0),
                                vMin + new Vector3(0, cs, 0),
                                vMin + new Vector3(cs, cs, 0),
                                vMin + new Vector3(0, 0, cs),
                                vMin + new Vector3(cs, 0, cs),
                                vMin + new Vector3(0, cs, cs),
                                vMin + new Vector3(cs, cs, cs)
                            }};

                            foreach (var corner in corners)
                            {{
                                lattice.AddBeam(vCenter, sr, corner, sr, true);
                            }}
                            lattice.AddSphere(vCenter, sr * 1.2f);
                        }}
                    }}
                }}

                Voxels voxLat = new Voxels(lattice);
                
                if (fShellThicknessMM > 0.001f)
                {{
                    Voxels innerVolume = new Voxels(voxels);
                    innerVolume.Offset(-fShellThicknessMM);

                    Voxels shell = new Voxels(voxels);
                    shell.BoolSubtract(innerVolume);

                    voxLat.BoolIntersect(innerVolume);

                    shell.BoolAdd(voxLat);
                    voxels = shell;
                }}
                else
                {{
                    voxels.BoolIntersect(voxLat);
                }}
            }}

            return new Mesh(voxels);
        }}

        public static Mesh LoadWavefrontObj(Library lib, string filePath)
        {{
            Mesh mesh = new Mesh(lib);
            using var reader = new StreamReader(filePath);
            string? line;

            while ((line = reader.ReadLine()) != null)
            {{
                line = line.Trim();
                if (line.Length == 0 || line.StartsWith("#"))
                    continue;

                if (line.StartsWith("v "))
                {{
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 4)
                    {{
                        float x = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        float y = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        float z = float.Parse(parts[3], CultureInfo.InvariantCulture);
                        mesh.nAddVertex(new Vector3(x, y, z));
                    }}
                }}
                else if (line.StartsWith("f "))
                {{
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 4)
                    {{
                        int totalVerts = mesh.nVertexCount();
                        int ParseIdx(string token)
                        {{
                            string idxStr = token.Split('/')[0];
                            int idx = int.Parse(idxStr, CultureInfo.InvariantCulture);
                            return (idx > 0) ? (idx - 1) : (totalVerts + idx);
                        }}

                        int i0 = ParseIdx(parts[1]);
                        int i1 = ParseIdx(parts[2]);
                        int i2 = ParseIdx(parts[3]);
                        mesh.nAddTriangle(i0, i1, i2);

                        for (int k = 4; k < parts.Length; k++)
                        {{
                            int prev = ParseIdx(parts[k - 1]);
                            int curr = ParseIdx(parts[k]);
                            mesh.nAddTriangle(i0, prev, curr);
                        }}
                    }}
                }}
            }}

            return mesh;
        }}

        public static void SaveToStl(Mesh mesh, string filePath)
        {{
            if (mesh == null)
                throw new ArgumentNullException(nameof(mesh), "Cannot export null mesh to STL.");

            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {{
                Directory.CreateDirectory(directory);
            }}

            mesh.SaveToStlFile(filePath);
        }}

        public static void SaveToObj(Mesh mesh, string filePath)
        {{
            if (mesh == null)
                throw new ArgumentNullException(nameof(mesh), "Cannot export null mesh to OBJ.");

            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {{
                Directory.CreateDirectory(directory);
            }}

            using var writer = new StreamWriter(filePath, false, System.Text.Encoding.UTF8);
            writer.WriteLine("# PicoGK Computational Engineering 3D Model Export");
            writer.WriteLine($"# Vertices: {{mesh.nVertexCount()}}, Triangles: {{mesh.nTriangleCount()}}");

            int vertexCount = mesh.nVertexCount();
            for (int i = 0; i < vertexCount; i++)
            {{
                Vector3 v = mesh.vecVertexAt(i);
                writer.WriteLine($"v {{v.X.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Y.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Z.ToString(\"F4\", CultureInfo.InvariantCulture)}}");
            }}

            int triangleCount = mesh.nTriangleCount();
            for (int i = 0; i < triangleCount; i++)
            {{
                Triangle tri = mesh.oTriangleAt(i);
                writer.WriteLine($"f {{tri.A + 1}} {{tri.B + 1}} {{tri.C + 1}}");
            }}
        }}
    }}
}}
"""


def _generate_ground_structure_csharp(params: GroundStructureTrussParams) -> str:
    """Generates standalone C# source code for an Optimal Ground Structure Truss in PicoGK."""
    beams_csharp = []
    for b in params.beams:
        r = max(b.radius, params.voxel_size_mm * 0.8)
        beams_csharp.append(
            f"                lattice.AddBeam(new Vector3({b.x1:.2f}f, {b.y1:.2f}f, {b.z1:.2f}f), {r:.3f}f, new Vector3({b.x2:.2f}f, {b.y2:.2f}f, {b.z2:.2f}f), {r:.3f}f, true);"
        )
        beams_csharp.append(
            f"                lattice.AddSphere(new Vector3({b.x1:.2f}f, {b.y1:.2f}f, {b.z1:.2f}f), {r * 1.15:.3f}f);"
        )
        beams_csharp.append(
            f"                lattice.AddSphere(new Vector3({b.x2:.2f}f, {b.y2:.2f}f, {b.z2:.2f}f), {r * 1.15:.3f}f);"
        )
    beams_block = "\n".join(beams_csharp)

    return f"""// ============================================================================
// PicoGK Computational Engineering - Optimal Generative Ground Structure Truss
// Generated: Auto-synthesized from Discrete Physics Equilibrium Optimization
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using PicoGK;

namespace PicoGK_Generated
{{
    public static class Program
    {{
        public static void Main(string[] args)
        {{
            float fVoxelSizeMM = {params.voxel_size_mm:.4f}f;
            string outputStl = args.Length > 0 ? args[0] : "GroundStructure_Output.stl";
            string outputObj = args.Length > 1 ? args[1] : "GroundStructure_Output.obj";

            Console.WriteLine("Initializing PicoGK OpenVDB Geometry Engine...");
            using var lib = new Library(fVoxelSizeMM);

            try
            {{
                Console.WriteLine("Generating Optimal Ground Structure Truss with Parameters:");
                Console.WriteLine(" - Active Beams Count: {len(params.beams)}");
                Console.WriteLine(" - Out-of-plane Depth: {params.depth_z_mm:.1f} mm");
                Console.WriteLine(" - Voxel Resolution: {params.voxel_size_mm:.3f} mm");

                var lattice = new Lattice(lib);

{beams_block}

                var voxels = new Voxels(lattice);
                Mesh mesh = new Mesh(voxels);

                Console.WriteLine($"Mesh generated successfully: {{mesh.nVertexCount()}} vertices, {{mesh.nTriangleCount()}} triangles.");

                SaveToStl(mesh, outputStl);
                SaveToObj(mesh, outputObj);
                Console.WriteLine($"Exported: {{outputStl}} & {{outputObj}}");
            }}
            catch (Exception ex)
            {{
                Console.Error.WriteLine($"[PICOGK_ERROR] {{ex.GetType().Name}}: {{ex.Message}}\\n{{ex.StackTrace}}");
                Environment.Exit(1);
            }}
        }}

        public static void SaveToStl(Mesh mesh, string filePath)
        {{
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory)) Directory.CreateDirectory(directory);
            mesh.SaveToStlFile(filePath);
        }}

        public static void SaveToObj(Mesh mesh, string filePath)
        {{
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory)) Directory.CreateDirectory(directory);

            using var writer = new StreamWriter(filePath, false, System.Text.Encoding.UTF8);
            writer.WriteLine("# PicoGK Generative Ground Structure 3D Export");
            writer.WriteLine($"# Vertices: {{mesh.nVertexCount()}}, Triangles: {{mesh.nTriangleCount()}}");

            int vertexCount = mesh.nVertexCount();
            for (int i = 0; i < vertexCount; i++)
            {{
                Vector3 v = mesh.vecVertexAt(i);
                writer.WriteLine($"v {{v.X.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Y.ToString(\"F4\", CultureInfo.InvariantCulture)}} {{v.Z.ToString(\"F4\", CultureInfo.InvariantCulture)}}");
            }}

            int triangleCount = mesh.nTriangleCount();
            for (int i = 0; i < triangleCount; i++)
            {{
                Triangle tri = mesh.oTriangleAt(i);
                writer.WriteLine($"f {{tri.A + 1}} {{tri.B + 1}} {{tri.C + 1}}");
            }}
        }}
    }}
}}
"""

