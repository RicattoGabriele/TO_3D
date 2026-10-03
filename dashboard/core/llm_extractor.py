"""
PicoGK Aerospace Component Generator - Dual-Path LLM & Deterministic NLP Extractor.
Extracts structured component parameters from natural language prompts using either
Google Gemini 2.5 Flash API or a deterministic zero-dependency heuristic parser.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from PicoGK_Dashboard.core.models import (
    ComponentParams,
    ComponentType,
    ConduitFlangeParams,
    LatticeBlockParams,
)

logger = logging.getLogger(__name__)


def _clean_json_markdown(text: str) -> str:
    """Strip markdown code fence blocks (```json ... ```) from LLM output."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def extract_parameters_gemini(
    prompt: str, api_key: Optional[str] = None
) -> Optional[ComponentParams]:
    """
    Extract parameters using Google Gemini API (gemini-2.5-flash).
    Requires google-genai SDK and valid API key.
    """
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_api_key:
        logger.info("No Gemini API key supplied or found in environment.")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_api_key)

        system_instruction = (
            "You are an expert aerospace computational engineer specializing in PicoGK additive manufacturing.\n"
            "Analyze the user's prompt and extract exact engineering parameters for one of the two templates:\n"
            "1. 'lattice_block': For lattice cubes, heat exchangers, TPMS Gyroid cores, BCC/FCC cellular blocks.\n"
            "   Fields: component_type ('lattice_block'), voxel_size_mm (float), size_x_mm (float), size_y_mm (float), "
            "size_z_mm (float), lattice_type (str, e.g. 'BCC', 'FCC', 'Gyroid'), cell_size_mm (float), "
            "strut_radius_mm (float), node_radius_mm (float), skin_thickness_mm (float), include_solid_plates (bool).\n"
            "2. 'conduit_flange': For pipes, tubes, propellant conduits, ducts with mounting flanges and bolt circles.\n"
            "   Fields: component_type ('conduit_flange'), voxel_size_mm (float), conduit_length_mm (float), "
            "outer_radius_mm (float), inner_radius_mm (float), flange_radius_mm (float), flange_thickness_mm (float), "
            "bolt_hole_count (int), bolt_circle_radius_mm (float), bolt_hole_radius_mm (float).\n\n"
            "Ensure all extracted dimensions satisfy physical boundaries (e.g. inner_radius < outer_radius, "
            "flange_radius > outer_radius, bolt_circle between outer_radius and flange_radius, strut_radius < cell_size * 0.45).\n"
            "Return ONLY a valid JSON object matching the chosen schema."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        if not response or not response.text:
            logger.warning("Empty response from Gemini API.")
            return None

        clean_text = _clean_json_markdown(response.text)
        data = json.loads(clean_text)

        comp_type = data.get("component_type", "").lower()
        if "lattice" in comp_type or "lattice_block" in comp_type:
            data["component_type"] = ComponentType.LATTICE_BLOCK
            return LatticeBlockParams(**data)
        elif "conduit" in comp_type or "flange" in comp_type or "pipe" in comp_type:
            data["component_type"] = ComponentType.CONDUIT_FLANGE
            return ConduitFlangeParams(**data)
        else:
            # Try parsing into either model
            try:
                data["component_type"] = ComponentType.LATTICE_BLOCK
                return LatticeBlockParams(**data)
            except ValidationError:
                data["component_type"] = ComponentType.CONDUIT_FLANGE
                return ConduitFlangeParams(**data)

    except Exception as ex:
        logger.warning("Gemini online extraction encountered an exception: %s. Falling back to offline NLP.", ex)
        return None


def _extract_flexible(patterns: List[str], text: str, default: Optional[float] = None) -> Optional[float]:
    """Helper regex extraction testing multiple prefix and suffix token patterns."""
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                for group in match.groups():
                    if group and re.match(r"^\d+(?:\.\d+)?$", group):
                        return float(group)
            except (ValueError, IndexError):
                continue
    return default


def _extract_int_flexible(patterns: List[str], text: str, default: Optional[int] = None) -> Optional[int]:
    """Helper regex extraction testing multiple integer token patterns."""
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                for group in match.groups():
                    if group and re.match(r"^\d+$", group):
                        return int(group)
            except (ValueError, IndexError):
                continue
    return default


def classify_template_type(prompt: str) -> ComponentType:
    """
    Classify whether the natural language prompt refers to a lattice block or conduit flange.
    """
    prompt_lower = prompt.lower()

    lattice_keywords = [
        "lattice", "gyroid", "bcc", "fcc", "tpms", "schwarz", "diamond",
        "infill", "cellular", "porous", "cell size", "strut", "cold plate",
        "heat sink", "heat exchanger", "cuboid", "lattice block"
    ]
    conduit_keywords = [
        "conduit", "flange", "pipe", "tube", "duct", "manifold", "bolt",
        "bore", "propellant", "hydraulic", "pcd", "pitch circle", "bolt circle"
    ]

    lattice_score = sum(1 for kw in lattice_keywords if kw in prompt_lower)
    conduit_score = sum(1 for kw in conduit_keywords if kw in prompt_lower)

    # Specific strong keywords
    if any(k in prompt_lower for k in ["flange", "conduit", "pipe", "bolt hole", "inner bore", "pcd", "mounting flange"]):
        conduit_score += 3
    if any(k in prompt_lower for k in ["lattice", "gyroid", "bcc", "fcc", "tpms", "strut", "cold plate"]):
        lattice_score += 3

    if conduit_score > lattice_score:
        return ComponentType.CONDUIT_FLANGE
    return ComponentType.LATTICE_BLOCK


def extract_parameters_offline(prompt: str) -> ComponentParams:
    """
    Comprehensive, deterministic, zero-dependency regex and heuristic NLP parser.
    Extracts all physical parameters with units, abbreviations, and contextual cues,
    filling smart defaults that satisfy physical boundary constraints.
    """
    template_type = classify_template_type(prompt)
    prompt_lower = prompt.lower()

    # Common extraction: Voxel size / pitch / resolution
    voxel_size = _extract_flexible(
        [
            r"(?:voxel(?:_size)?|pitch|resolution|res)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
            r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:voxel(?:_size)?|pitch|resolution|res)",
        ],
        prompt_lower,
        default=0.5
    )
    if voxel_size is None or voxel_size <= 0:
        voxel_size = 0.5

    if template_type == ComponentType.LATTICE_BLOCK:
        # 1. 3D Bounding Dimensions (e.g. 50x50x50, 40 x 40 x 60 mm, 50 by 50 by 50, 40x40x40mm)
        dim_3d = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:x|×|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:x|×|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
            prompt_lower
        )
        if dim_3d:
            size_x = float(dim_3d.group(1))
            size_y = float(dim_3d.group(2))
            size_z = float(dim_3d.group(3))
        else:
            # 2D or 1D cube shorthand (e.g. 40mm cube, 50x50 cube)
            dim_2d = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:x|×|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:mm)?", prompt_lower)
            cube_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:cube|block)", prompt_lower)
            if dim_2d:
                size_x = float(dim_2d.group(1))
                size_y = float(dim_2d.group(2))
                size_z = _extract_flexible(
                    [
                        r"(?:height|depth|size_z|z)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)",
                        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:height|depth|high|thick)",
                    ],
                    prompt_lower,
                    default=size_x
                )
            elif cube_match:
                size_x = float(cube_match.group(1))
                size_y = size_x
                size_z = size_x
            else:
                size_x = _extract_flexible(
                    [
                        r"(?:size_x|length|width|dim_x|x)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)",
                        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:length|width|long)",
                    ],
                    prompt_lower,
                    default=40.0
                )
                size_y = _extract_flexible(
                    [
                        r"(?:size_y|width|depth|dim_y|y)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)",
                        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:width|depth|wide)",
                    ],
                    prompt_lower,
                    default=size_x
                )
                size_z = _extract_flexible(
                    [
                        r"(?:size_z|height|depth|dim_z|z)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)",
                        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:height|high|thick)",
                    ],
                    prompt_lower,
                    default=size_x
                )

        min_dim = min(size_x, size_y, size_z)

        # 2. Lattice Type
        lattice_type = "BCC"
        if "gyroid" in prompt_lower:
            lattice_type = "Gyroid"
        elif "fcc" in prompt_lower:
            lattice_type = "FCC"
        elif "diamond" in prompt_lower:
            lattice_type = "Diamond"
        elif "schwarz" in prompt_lower:
            lattice_type = "SchwarzPrimitive"
        elif "bcc" in prompt_lower:
            lattice_type = "BCC"

        # 3. Cell Size
        cell_size = _extract_flexible(
            [
                r"(?:cell(?:_size)?|unit_cell|period|repeat)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:cell(?:_size)?|unit_cell|period|repeat|cells)",
            ],
            prompt_lower,
            default=min(10.0, min_dim / 2.0)
        )
        if cell_size is None or cell_size <= 0:
            cell_size = min(10.0, min_dim / 2.0)
        if cell_size > min_dim:
            cell_size = max(1.0, min_dim / 2.0)

        # 4. Strut radius / Wall thickness
        strut_radius = _extract_flexible(
            [
                r"(?:strut(?:_radius)?|struts?|beam|wall(?:_thickness)?|thickness)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:strut(?:_radius)?|struts?|beam|wall(?:_thickness)?|thickness|wall)",
            ],
            prompt_lower,
            default=min(1.0, cell_size * 0.15)
        )
        strut_diam = _extract_flexible(
            [
                r"(?:strut_diameter|strut\s+dia(?:meter)?)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:strut\s+dia(?:meter)?|strut_diameter)",
            ],
            prompt_lower
        )
        if strut_diam is not None and strut_diam > 0:
            strut_radius = strut_diam / 2.0

        if strut_radius is None or strut_radius <= 0:
            strut_radius = max(0.2, min(1.0, cell_size * 0.15))
        if strut_radius >= cell_size * 0.45:
            strut_radius = cell_size * 0.2

        # 5. Node radius
        node_radius = _extract_flexible(
            [
                r"(?:node(?:_radius)?|nodes?|joint)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:node(?:_radius)?|nodes?|joint)",
            ],
            prompt_lower,
            default=max(strut_radius * 1.5, strut_radius + 0.5)
        )
        if node_radius is None or node_radius < strut_radius:
            node_radius = max(strut_radius * 1.5, strut_radius + 0.3)

        # 6. Skin thickness
        skin_thickness = _extract_flexible(
            [
                r"(?:skin(?:_thickness)?|skin|plate_thickness|shell)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:skin(?:_thickness)?|skin|plate_thickness|shell)",
            ],
            prompt_lower,
            default=min(2.0, min_dim * 0.1)
        )
        if "no skin" in prompt_lower or "open lattice" in prompt_lower or "without skin" in prompt_lower:
            skin_thickness = 0.0
        if skin_thickness is None:
            skin_thickness = 2.0
        if skin_thickness >= min_dim / 2.0:
            skin_thickness = max(0.0, min_dim * 0.05)

        # 7. Solid plates
        include_solid_plates = True
        if any(neg in prompt_lower for neg in ["no plate", "without plate", "open frame", "open top", "no mounting plate"]):
            include_solid_plates = False

        # 8. Adjust voxel size if too coarse for strut
        if voxel_size > strut_radius * 2.0:
            voxel_size = max(0.1, strut_radius * 0.8)

        return LatticeBlockParams(
            component_type=ComponentType.LATTICE_BLOCK,
            voxel_size_mm=voxel_size,
            size_x_mm=size_x,
            size_y_mm=size_y,
            size_z_mm=size_z,
            lattice_type=lattice_type,
            cell_size_mm=cell_size,
            strut_radius_mm=strut_radius,
            node_radius_mm=node_radius,
            skin_thickness_mm=skin_thickness,
            include_solid_plates=include_solid_plates,
        )

    else:
        # Conduit & Flange extraction
        # 1. Conduit length
        conduit_length = _extract_flexible(
            [
                r"(?:conduit_length|pipe_length|total_length|length|height)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:conduit_length|pipe_length|total_length|length|height|long)",
            ],
            prompt_lower,
            default=60.0
        )
        if conduit_length is None or conduit_length <= 0:
            conduit_length = 60.0

        # 2. Outer radius / diameter
        outer_radius = _extract_flexible(
            [
                r"(?:outer_radius|outer_rad|pipe_radius|tube_radius|pipe_rad)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:outer_radius|outer_rad|pipe_radius|tube_radius)",
            ],
            prompt_lower
        )
        if outer_radius is None:
            outer_diam = _extract_flexible(
                [
                    r"(?:outer\s*(?:pipe\s*)?diameter|pipe\s*diameter|outer\s*dia(?:meter)?|pipe\s*od|od|diameter)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:outer\s*(?:pipe\s*)?diameter|pipe\s*diameter|outer\s*dia(?:meter)?|pipe\s*od|od)",
                ],
                prompt_lower
            )
            if outer_diam:
                outer_radius = outer_diam / 2.0
            else:
                outer_radius = 15.0

        # 3. Inner radius / diameter / bore
        inner_radius = _extract_flexible(
            [
                r"(?:inner\s*(?:bore\s*)?radius|inner_rad|bore_radius|bore_rad)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:inner\s*(?:bore\s*)?radius|inner_rad|bore_radius|bore_rad)",
            ],
            prompt_lower
        )
        if inner_radius is None:
            inner_diam = _extract_flexible(
                [
                    r"(?:inner\s*(?:bore\s*)?diameter|bore\s*diameter|inner\s*dia(?:meter)?|bore\s*dia(?:meter)?|inner\s*bore|id)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:inner\s*(?:bore\s*)?diameter|bore\s*diameter|inner\s*dia(?:meter)?|bore\s*dia(?:meter)?|inner\s*bore|id)",
                    r"\bbore\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                ],
                prompt_lower
            )
            if inner_diam:
                inner_radius = inner_diam / 2.0
            else:
                # Check if wall thickness was specified
                wall_th = _extract_flexible(
                    [
                        r"(?:wall_thickness|wall|pipe_wall)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:wall_thickness|wall|pipe_wall)",
                    ],
                    prompt_lower
                )
                if wall_th and wall_th < outer_radius:
                    inner_radius = outer_radius - wall_th
                else:
                    inner_radius = outer_radius * 0.67

        # Ensure inner < outer
        if inner_radius >= outer_radius:
            inner_radius = outer_radius * 0.67

        # 4. Flange radius / diameter
        flange_radius = _extract_flexible(
            [
                r"(?:flange\s*(?:outer\s*)?radius|flange_radius|flange_rad)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:flange\s*(?:outer\s*)?radius|flange_radius|flange_rad)",
            ],
            prompt_lower
        )
        if flange_radius is None:
            flange_diam = _extract_flexible(
                [
                    r"(?:flange\s*(?:outer\s*)?diameter|flange\s*dia(?:meter)?|flange\s*od|flange\s*size|flange\s*disc)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:flange\s*(?:outer\s*)?diameter|flange\s*dia(?:meter)?|flange\s*od|flange\s*size)",
                ],
                prompt_lower
            )
            if flange_diam:
                flange_radius = flange_diam / 2.0
            else:
                flange_radius = max(30.0, outer_radius * 2.0)

        if flange_radius <= outer_radius:
            flange_radius = outer_radius + 15.0

        # 5. Flange thickness
        flange_thickness = _extract_flexible(
            [
                r"(?:flange\s*thickness|flange\s*thick(?:ness)?|flange\s*depth)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:flange\s*thickness|flange\s*thick(?:ness)?|flange\s*depth)",
            ],
            prompt_lower,
            default=min(8.0, conduit_length * 0.2)
        )
        if flange_thickness is None or flange_thickness <= 0:
            flange_thickness = min(8.0, conduit_length * 0.2)
        if flange_thickness > conduit_length:
            flange_thickness = conduit_length * 0.25

        # 6. Bolt hole count
        bolt_count = _extract_int_flexible(
            [
                r"(\d+)\s*(?:-?\s*bolt|bolt\s*hole|holes?|bolts?)\b",
                r"(?:bolt_hole_count|bolt_count|num_bolts|bolts)\s*(?:of|:|=)?\s*(\d+)",
                r"(\d+)\s*x\s*(?:m\d+|bolt)",
            ],
            prompt_lower,
            default=6
        )
        if "no bolt" in prompt_lower or "without bolt" in prompt_lower or "0 bolt" in prompt_lower:
            bolt_count = 0
        if bolt_count is None or bolt_count < 0:
            bolt_count = 6

        # 7. Bolt hole radius
        bolt_hole_radius = _extract_flexible(
            [
                r"(?:bolt\s*(?:hole\s*)?radius|hole_radius)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:bolt\s*(?:hole\s*)?radius|hole_radius)",
            ],
            prompt_lower
        )
        if bolt_hole_radius is None:
            bolt_hole_diam = _extract_flexible(
                [
                    r"(?:bolt\s*(?:hole\s*)?diameter|hole_diameter|bolt\s*dia)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:bolt\s*(?:hole\s*)?diameter|hole_diameter|bolt\s*dia)",
                ],
                prompt_lower
            )
            m_bolt = re.search(r"\bm(\d+(?:\.\d+)?)\b", prompt_lower)
            if bolt_hole_diam:
                bolt_hole_radius = bolt_hole_diam / 2.0
            elif m_bolt:
                bolt_hole_radius = (float(m_bolt.group(1)) + 0.5) / 2.0
            else:
                bolt_hole_radius = 2.5

        # 8. Bolt circle radius / Pitch Circle Diameter (PCD)
        bolt_circle_radius = _extract_flexible(
            [
                r"(?:bolt\s*circle\s*radius|bolt_circle_rad|bolt_circle|pitch_circle_radius|pcr)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:bolt\s*circle\s*radius|bolt_circle_rad|bolt_circle|pitch_circle_radius|pcr)",
            ],
            prompt_lower
        )
        if bolt_circle_radius is None:
            pcd = _extract_flexible(
                [
                    r"(?:bolt\s*circle\s*diameter|pitch\s*circle\s*diameter|pcd|bcd)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:\w+\s+)?(?:bolt\s*circle\s*diameter|pitch\s*circle\s*diameter|pcd|bcd)",
                    r"(?:on|at)\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:pcd|bcd|bolt\s*circle)",
                ],
                prompt_lower
            )
            if pcd:
                bolt_circle_radius = pcd / 2.0
            else:
                bolt_circle_radius = (outer_radius + flange_radius) / 2.0

        # Adjust bolt dimensions to guarantee physical clearance
        if bolt_count > 0:
            available_flange_space = flange_radius - outer_radius
            if bolt_hole_radius * 2.5 > available_flange_space:
                bolt_hole_radius = max(1.0, available_flange_space * 0.25)

            # Center bolt circle midway between pipe OD and flange edge
            min_bc = outer_radius + bolt_hole_radius + 1.0
            max_bc = flange_radius - bolt_hole_radius - 1.0

            if min_bc >= max_bc:
                # Expand flange radius to make space
                flange_radius = outer_radius + (bolt_hole_radius * 4.0) + 4.0
                min_bc = outer_radius + bolt_hole_radius + 1.0
                max_bc = flange_radius - bolt_hole_radius - 1.0

            if bolt_circle_radius <= min_bc or bolt_circle_radius >= max_bc:
                bolt_circle_radius = (outer_radius + flange_radius) / 2.0

            # Verify no adjacent bolt overlap
            if bolt_count >= 2:
                import math
                angle_step = 2.0 * math.pi / bolt_count
                chord = 2.0 * bolt_circle_radius * math.sin(angle_step / 2.0)
                if chord <= 2.0 * bolt_hole_radius:
                    bolt_hole_radius = max(0.8, (chord * 0.4))

        # Adjust voxel size if pipe wall is thin
        wall_th = outer_radius - inner_radius
        if voxel_size > wall_th * 1.5:
            voxel_size = max(0.1, wall_th * 0.5)

        return ConduitFlangeParams(
            component_type=ComponentType.CONDUIT_FLANGE,
            voxel_size_mm=voxel_size,
            conduit_length_mm=conduit_length,
            outer_radius_mm=outer_radius,
            inner_radius_mm=inner_radius,
            flange_radius_mm=flange_radius,
            flange_thickness_mm=flange_thickness,
            bolt_hole_count=bolt_count,
            bolt_circle_radius_mm=bolt_circle_radius,
            bolt_hole_radius_mm=bolt_hole_radius,
        )


def extract_parameters(
    prompt: str, api_key: Optional[str] = None
) -> ComponentParams:
    """
    Unified entrypoint for NLP parameter extraction.
    Attempts online Gemini extraction first if API key is provided or configured in env;
    falls back reliably to deterministic regex & heuristic parser.
    """
    if not prompt or not prompt.strip():
        # Default fallback to a standard lattice block
        return LatticeBlockParams()

    # Try online extraction if API key is present
    has_key = bool(api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if has_key:
        try:
            result = extract_parameters_gemini(prompt, api_key=api_key)
            if result is not None:
                return result
        except Exception as ex:
            logger.warning("Online extraction failed, proceeding with offline parser: %s", ex)

    # Deterministic offline parser
    return extract_parameters_offline(prompt)
