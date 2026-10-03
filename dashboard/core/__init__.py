"""
PicoGK Aerospace Component Generator - Core Module.
Exposes models, NLP extractors, template engines, and schemas.
"""

from PicoGK_Dashboard.core.models import (
    ComponentParams,
    ComponentType,
    ConduitFlangeParams,
    LatticeBlockParams,
)
from PicoGK_Dashboard.core.llm_extractor import (
    classify_template_type,
    extract_parameters,
    extract_parameters_gemini,
    extract_parameters_offline,
)
from PicoGK_Dashboard.core.template_engine import (
    generate_csharp_source,
    write_params_json,
)

__all__ = [
    "ComponentType",
    "LatticeBlockParams",
    "ConduitFlangeParams",
    "ComponentParams",
    "extract_parameters",
    "extract_parameters_gemini",
    "extract_parameters_offline",
    "classify_template_type",
    "write_params_json",
    "generate_csharp_source",
]
