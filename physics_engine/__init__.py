"""
Physics Engine Package - 2D & 3D Multi-Objective Continuum Topology Optimization.
"""

from .ground_structure import GroundStructure2D, Member2D, Node2D, GroundStructureResult
from .simp_engine_2d import SIMPOptimizer2D, SIMPResult
from .simp_engine_3d import SIMPOptimizer3D, SIMPResult3D

__all__ = [
    "GroundStructure2D",
    "Member2D",
    "Node2D",
    "GroundStructureResult",
    "SIMPOptimizer2D",
    "SIMPResult",
    "SIMPOptimizer3D",
    "SIMPResult3D",
]
