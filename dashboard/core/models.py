"""
PicoGK Aerospace Component Generator - Pydantic Parameter Models & Boundary Validators.
Defines strongly-typed schemas and strict aerospace physical constraint validations.
"""

from enum import Enum
import math
import os
from typing import Literal, Union
from pydantic import BaseModel, Field, model_validator


class ComponentType(str, Enum):
    """Supported aerospace parametric component types."""
    LATTICE_BLOCK = "lattice_block"
    CONDUIT_FLANGE = "conduit_flange"
    IMPORTED_MESH = "imported_mesh"
    GROUND_STRUCTURE = "ground_structure"


class LatticeBlockParams(BaseModel):
    """
    Parameters for parametric lightweight lattice blocks (e.g. BCC, FCC, TPMS Gyroid).
    Enforces strict physical and manufacturing boundary constraints.
    """
    component_type: Literal[ComponentType.LATTICE_BLOCK, "lattice_block"] = ComponentType.LATTICE_BLOCK
    voxel_size_mm: float = Field(
        default=0.5,
        ge=0.05,
        le=10.0,
        description="Voxel pitch / grid resolution in mm"
    )
    size_x_mm: float = Field(
        default=40.0,
        ge=5.0,
        le=500.0,
        description="Bounding dimension along X in mm"
    )
    size_y_mm: float = Field(
        default=40.0,
        ge=5.0,
        le=500.0,
        description="Bounding dimension along Y in mm"
    )
    size_z_mm: float = Field(
        default=40.0,
        ge=5.0,
        le=500.0,
        description="Bounding dimension along Z in mm"
    )
    lattice_type: str = Field(
        default="BCC",
        description="Lattice unit cell topology (e.g. BCC, FCC, Gyroid, SchwarzPrimitive, Diamond)"
    )
    cell_size_mm: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Repeat unit cell dimension in mm"
    )
    strut_radius_mm: float = Field(
        default=1.0,
        ge=0.1,
        le=25.0,
        description="Strut beam radius or TPMS wall half-thickness in mm"
    )
    node_radius_mm: float = Field(
        default=1.5,
        ge=0.1,
        le=30.0,
        description="Spherical node reinforcement radius in mm"
    )
    skin_thickness_mm: float = Field(
        default=2.0,
        ge=0.0,
        le=50.0,
        description="Solid boundary skin thickness in mm (0 for open lattice)"
    )
    include_solid_plates: bool = Field(
        default=True,
        description="Whether to include top and bottom mounting plates / solid skins"
    )

    @model_validator(mode="after")
    def validate_lattice_geometry(self) -> "LatticeBlockParams":
        """Validate physical and geometric compatibility for lattice block."""
        min_dim = min(self.size_x_mm, self.size_y_mm, self.size_z_mm)

        if self.cell_size_mm > min_dim:
            raise ValueError(
                f"Cell size ({self.cell_size_mm:.2f}mm) cannot exceed minimum block dimension ({min_dim:.2f}mm)."
            )

        if self.strut_radius_mm >= self.cell_size_mm * 0.45:
            raise ValueError(
                f"Strut radius ({self.strut_radius_mm:.2f}mm) must be less than 45% of cell size "
                f"({self.cell_size_mm:.2f}mm) to avoid solid clogging."
            )

        if self.node_radius_mm < self.strut_radius_mm:
            raise ValueError(
                f"Node radius ({self.node_radius_mm:.2f}mm) must be greater than or equal to strut radius "
                f"({self.strut_radius_mm:.2f}mm)."
            )

        if self.skin_thickness_mm >= min_dim / 2.0:
            raise ValueError(
                f"Skin thickness ({self.skin_thickness_mm:.2f}mm) must be strictly less than half of "
                f"minimum block dimension ({min_dim / 2.0:.2f}mm)."
            )

        if self.voxel_size_mm > self.strut_radius_mm * 2.0:
            raise ValueError(
                f"Voxel size ({self.voxel_size_mm:.2f}mm) is too coarse for strut radius ({self.strut_radius_mm:.2f}mm); "
                f"must be <= 2 * strut_radius ({self.strut_radius_mm * 2.0:.2f}mm)."
            )

        return self


class ConduitFlangeParams(BaseModel):
    """
    Parameters for parametric hydraulic conduits and bolted mounting flanges.
    Enforces strict physical containment, clearance, and non-overlap constraints.
    """
    component_type: Literal[ComponentType.CONDUIT_FLANGE, "conduit_flange"] = ComponentType.CONDUIT_FLANGE
    voxel_size_mm: float = Field(
        default=0.5,
        ge=0.05,
        le=10.0,
        description="Voxel pitch / grid resolution in mm"
    )
    conduit_length_mm: float = Field(
        default=60.0,
        ge=5.0,
        le=1000.0,
        description="Total conduit pipe length in mm"
    )
    outer_radius_mm: float = Field(
        default=15.0,
        ge=2.0,
        le=500.0,
        description="Outer radius of the conduit pipe in mm"
    )
    inner_radius_mm: float = Field(
        default=10.0,
        ge=1.0,
        le=490.0,
        description="Inner bore radius of the conduit pipe in mm"
    )
    flange_radius_mm: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="Outer radius of the mounting flange disc in mm"
    )
    flange_thickness_mm: float = Field(
        default=8.0,
        ge=1.0,
        le=200.0,
        description="Thickness of the mounting flange in mm"
    )
    bolt_hole_count: int = Field(
        default=6,
        ge=0,
        le=64,
        description="Number of bolt holes on the flange"
    )
    bolt_circle_radius_mm: float = Field(
        default=23.0,
        ge=2.0,
        le=550.0,
        description="Pitch circle radius for bolt holes in mm"
    )
    bolt_hole_radius_mm: float = Field(
        default=2.5,
        ge=0.5,
        le=50.0,
        description="Radius of each bolt hole in mm"
    )

    @model_validator(mode="after")
    def validate_conduit_physics(self) -> "ConduitFlangeParams":
        """Validate physical and geometric compatibility for conduit and flange."""
        if self.inner_radius_mm >= self.outer_radius_mm:
            raise ValueError(
                f"Inner radius ({self.inner_radius_mm:.2f}mm) must be strictly less than "
                f"outer radius ({self.outer_radius_mm:.2f}mm)."
            )

        if self.flange_radius_mm <= self.outer_radius_mm:
            raise ValueError(
                f"Flange radius ({self.flange_radius_mm:.2f}mm) must be strictly greater than "
                f"conduit outer radius ({self.outer_radius_mm:.2f}mm)."
            )

        if self.flange_thickness_mm > self.conduit_length_mm:
            raise ValueError(
                f"Flange thickness ({self.flange_thickness_mm:.2f}mm) cannot exceed "
                f"total conduit length ({self.conduit_length_mm:.2f}mm)."
            )

        wall_thickness = self.outer_radius_mm - self.inner_radius_mm
        if self.voxel_size_mm > wall_thickness * 1.5:
            raise ValueError(
                f"Voxel size ({self.voxel_size_mm:.2f}mm) is too coarse for pipe wall thickness ({wall_thickness:.2f}mm)."
            )

        if self.bolt_hole_count > 0:
            # Bolt holes must clear pipe outer surface
            inner_bolt_clearance = self.bolt_circle_radius_mm - self.bolt_hole_radius_mm
            if inner_bolt_clearance <= self.outer_radius_mm:
                raise ValueError(
                    f"Bolt holes (inner clearance {inner_bolt_clearance:.2f}mm) cut into conduit outer wall "
                    f"({self.outer_radius_mm:.2f}mm). Increase bolt_circle_radius_mm or reduce bolt_hole_radius_mm."
                )

            # Bolt holes must remain within outer flange boundary
            outer_bolt_boundary = self.bolt_circle_radius_mm + self.bolt_hole_radius_mm
            if outer_bolt_boundary >= self.flange_radius_mm:
                raise ValueError(
                    f"Bolt holes (outer boundary {outer_bolt_boundary:.2f}mm) exceed flange radius "
                    f"({self.flange_radius_mm:.2f}mm). Reduce bolt_circle_radius_mm or enlarge flange_radius_mm."
                )

            # Adjacent bolt holes must not overlap
            if self.bolt_hole_count >= 2:
                # Chord distance between bolt hole centers on circle
                angle_step = 2.0 * math.pi / self.bolt_hole_count
                chord_distance = 2.0 * self.bolt_circle_radius_mm * math.sin(angle_step / 2.0)
                min_required_distance = 2.0 * self.bolt_hole_radius_mm
                if chord_distance <= min_required_distance:
                    raise ValueError(
                        f"Bolt holes overlap along pitch circle. Chord distance between centers is "
                        f"{chord_distance:.2f}mm, but requires > {min_required_distance:.2f}mm for "
                        f"{self.bolt_hole_count} holes of radius {self.bolt_hole_radius_mm:.2f}mm."
                    )

        return self



class ImportedMeshParams(BaseModel):
    """
    Parameters for external CAD / topologically optimized mesh post-processing
    (e.g. Altair Inspire STL/OBJ), including voxel thickening/shrinking and internal lattice infill.
    """
    component_type: Literal[ComponentType.IMPORTED_MESH, "imported_mesh"] = ComponentType.IMPORTED_MESH
    mesh_file_path: str = Field(
        ...,
        description="Absolute or relative filesystem path to the source STL or OBJ mesh file"
    )
    voxel_size_mm: float = Field(
        default=0.5,
        ge=0.05,
        le=10.0,
        description="Voxel pitch / OpenVDB grid resolution in mm"
    )
    offset_thickness_mm: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description="Offset distance in mm (positive to thicken/expand, negative to shrink/erode, 0 for identity)"
    )
    apply_lattice_infill: bool = Field(
        default=False,
        description="Whether to generate and boolean intersect internal lattice infill with the mesh volume"
    )
    lattice_type: str = Field(
        default="GRID",
        description="Lattice unit cell topology (e.g. GRID, Gyroid, BCC, FCC, SchwarzPrimitive, Diamond)"
    )
    cell_size_mm: float = Field(
        default=5.0,
        ge=1.0,
        le=50.0,
        description="Repeat unit cell dimension in mm"
    )
    strut_radius_mm: float = Field(
        default=0.5,
        ge=0.05,
        le=10.0,
        description="Strut beam radius or TPMS wall half-thickness in mm"
    )
    shell_thickness_mm: float = Field(
        default=1.0,
        ge=0.0,
        le=20.0,
        description="Shell thickness in mm to preserve the external contour when infill is applied"
    )

    @model_validator(mode="after")
    def validate_imported_mesh_physics(self) -> "ImportedMeshParams":
        """Validate file path integrity and geometric compatibility for imported mesh processing."""
        p_str = self.mesh_file_path.strip()
        if not p_str:
            raise ValueError("Mesh file path cannot be empty.")

        ext = os.path.splitext(p_str)[1].lower()
        if ext not in [".stl", ".obj"]:
            raise ValueError(f"Mesh file must have .stl or .obj extension, got '{ext}'.")

        if self.apply_lattice_infill:
            if self.strut_radius_mm >= self.cell_size_mm * 0.45:
                raise ValueError(
                    f"Strut radius ({self.strut_radius_mm:.2f}mm) must be less than 45% of cell size "
                    f"({self.cell_size_mm:.2f}mm) to avoid solid clogging."
                )

            if self.voxel_size_mm > self.strut_radius_mm * 2.0:
                raise ValueError(
                    f"Voxel size ({self.voxel_size_mm:.2f}mm) is too coarse for strut radius ({self.strut_radius_mm:.2f}mm); "
                    f"must be <= 2 * strut_radius ({self.strut_radius_mm * 2.0:.2f}mm)."
                )

        return self


class BeamParam(BaseModel):
    """Geometric parameter for a single truss beam."""
    x1: float
    y1: float
    z1: float = 0.0
    x2: float
    y2: float
    z2: float = 0.0
    radius: float = Field(default=0.5, ge=0.01, le=50.0)
    force: float = 0.0


class GroundStructureTrussParams(BaseModel):
    """
    Parameters for 3D voxel synthesis of optimal Ground Structure trusses in PicoGK.
    """
    component_type: Literal[ComponentType.GROUND_STRUCTURE, "ground_structure"] = ComponentType.GROUND_STRUCTURE
    voxel_size_mm: float = Field(
        default=0.5,
        ge=0.05,
        le=5.0,
        description="Voxel pitch / OpenVDB grid resolution in mm"
    )
    depth_z_mm: float = Field(
        default=10.0,
        ge=0.1,
        le=200.0,
        description="Arbitrary 3D out-of-plane thickness/depth for 2D truss in mm"
    )
    beams: list[BeamParam] = Field(
        default_factory=list,
        description="List of active structural members with endpoints and computed radii"
    )

    @model_validator(mode="after")
    def validate_truss_physics(self) -> "GroundStructureTrussParams":
        if not self.beams:
            raise ValueError("Ground structure must have at least one beam.")
        return self


# Union type of all valid component parameter models
ComponentParams = Union[LatticeBlockParams, ConduitFlangeParams, ImportedMeshParams, GroundStructureTrussParams]

