"""
PicoGK Aerospace Component Generator - Subprocess Execution Runner.
Orchestrates background .NET PicoGK engine invocations, parameters injection,
real-time stdout/stderr capture, timeout guards, and structured result extraction.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any, List, Optional, Union
from pydantic import BaseModel

from PicoGK_Dashboard.core.models import (
    ComponentParams,
    ComponentType,
    ConduitFlangeParams,
    ImportedMeshParams,
    LatticeBlockParams,
)
from PicoGK_Dashboard.core.template_engine import (
    generate_csharp_source,
    write_params_json,
)


@dataclass
class GenerationResult:
    """Structured result returned by the PicoGK execution pipeline."""
    success: bool
    stl_path: Optional[str] = None
    obj_path: Optional[str] = None
    cs_path: Optional[str] = None
    vertex_count: int = 0
    triangle_count: int = 0
    execution_time_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)


def find_dotnet_executable() -> Optional[str]:
    """
    Locates a working .NET CLI executable on the system.
    Searches user-profile dotnet install, PATH, and standard program files.
    """
    # 1. Check USERPROFILE\.dotnet\dotnet.exe (preferred for local SDKs like .NET 9)
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        user_dotnet = os.path.join(user_profile, ".dotnet", "dotnet.exe")
        if os.path.exists(user_dotnet):
            return user_dotnet

    # 2. Check DOTNET_ROOT if set
    dotnet_root = os.environ.get("DOTNET_ROOT")
    if dotnet_root:
        root_dotnet = os.path.join(dotnet_root, "dotnet.exe" if os.name == "nt" else "dotnet")
        if os.path.exists(root_dotnet):
            return root_dotnet

    # 3. Check system PATH
    which_dotnet = shutil.which("dotnet")
    if which_dotnet and os.path.exists(which_dotnet):
        return which_dotnet

    # 4. Standard Windows Program Files location
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf_dotnet = os.path.join(program_files, "dotnet", "dotnet.exe")
    if os.path.exists(pf_dotnet):
        return pf_dotnet

    return None


def get_dotnet_environment() -> dict:
    """
    Prepares an environment dict with DOTNET_ROOT and PATH correctly configured
    for .NET 9.0 SDK discovery.
    """
    env = os.environ.copy()
    user_profile = os.environ.get("USERPROFILE")

    if user_profile:
        user_dotnet_dir = os.path.join(user_profile, ".dotnet")
        if os.path.exists(os.path.join(user_dotnet_dir, "dotnet.exe")):
            env["DOTNET_ROOT"] = user_dotnet_dir
            current_path = env.get("PATH", "")
            if user_dotnet_dir not in current_path:
                env["PATH"] = f"{user_dotnet_dir};{current_path}"
            return env

    dotnet_exe = find_dotnet_executable()
    if dotnet_exe:
        dotnet_dir = os.path.dirname(dotnet_exe)
        env["DOTNET_ROOT"] = dotnet_dir
        current_path = env.get("PATH", "")
        if dotnet_dir not in current_path:
            env["PATH"] = f"{dotnet_dir};{current_path}"

    return env


def get_csharp_project_path() -> Path:
    """Returns the absolute path to PicoGK_Engine.csproj."""
    current_dir = Path(__file__).resolve().parent
    # Check PicoGK_Dashboard/csharp/PicoGK_Engine.csproj
    dashboard_dir = current_dir.parent
    csproj_path = dashboard_dir / "csharp" / "PicoGK_Engine.csproj"
    if csproj_path.exists():
        return csproj_path

    # Fallback to root search
    root_dir = dashboard_dir.parent
    csproj_root_path = root_dir / "PicoGK_Dashboard" / "csharp" / "PicoGK_Engine.csproj"
    if csproj_root_path.exists():
        return csproj_root_path

    return csproj_path


def run_generation(
    params: Union[BaseModel, Any],
    output_dir: str,
    timeout_sec: int = 60,
    output_format: str = "all",
    mesh_basename: str = "model"
) -> GenerationResult:
    """
    Executes the PicoGK C# geometry engine in a background subprocess.

    Args:
        params: Validated Pydantic model (LatticeBlockParams, ConduitFlangeParams) or dict/object.
        output_dir: Output directory where STL, OBJ, and C# artifacts will be written.
        timeout_sec: Maximum execution timeout in seconds.
        output_format: "stl", "obj", or "all" (both STL & OBJ).
        mesh_basename: Base name for exported mesh files.

    Returns:
        GenerationResult containing execution status, artifact paths, geometry metrics,
        and full diagnostics.
    """
    start_time = time.time()
    out_dir_p = Path(output_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # 1. Determine template name and extract parameters safely
    template_name = ""
    voxel_size_mm = 0.5

    try:
        if hasattr(params, "component_type"):
            ct = getattr(params, "component_type")
            template_name = ct.value if hasattr(ct, "value") else str(ct)
        elif isinstance(params, dict) and "component_type" in params:
            template_name = str(params["component_type"])

        if hasattr(params, "voxel_size_mm"):
            voxel_size_mm = float(getattr(params, "voxel_size_mm"))
        elif isinstance(params, dict) and "voxel_size_mm" in params:
            voxel_size_mm = float(params["voxel_size_mm"])
    except Exception as e:
        return GenerationResult(
            success=False,
            execution_time_sec=time.time() - start_time,
            error_message=f"Failed to parse parameter model: {str(e)}",
            stderr=traceback.format_exc()
        )

    if not template_name or template_name not in ["lattice_block", "conduit_flange", "imported_mesh", "ground_structure"]:
        return GenerationResult(
            success=False,
            execution_time_sec=time.time() - start_time,
            error_message=f"Unknown or unsupported component template: '{template_name}'. Expected 'lattice_block', 'conduit_flange', 'imported_mesh', or 'ground_structure'.",
            stderr=f"[PICOGK_ERROR] ArgumentException: Unknown template '{template_name}'"
        )

    # 2. Write params.json
    params_json_path = out_dir_p / "params.json"
    try:
        if isinstance(params, BaseModel):
            write_params_json(params, str(params_json_path))
        elif isinstance(params, dict):
            with open(params_json_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
        else:
            # Fallback reflection dump
            data = {k: v for k, v in params.__dict__.items() if not k.startswith("_")}
            with open(params_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        return GenerationResult(
            success=False,
            execution_time_sec=time.time() - start_time,
            error_message=f"Failed to serialize parameters to JSON: {str(e)}",
            stderr=traceback.format_exc()
        )

    # 3. Synthesize standalone C# code for user inspection / download
    cs_path = out_dir_p / "GeneratedModel.cs"
    try:
        if isinstance(params, BaseModel):
            generate_csharp_source(params, str(cs_path))
        else:
            # Try to cast to typed model if possible
            if template_name == "lattice_block":
                typed_p = LatticeBlockParams(**(params if isinstance(params, dict) else params.__dict__))
                generate_csharp_source(typed_p, str(cs_path))
            elif template_name == "conduit_flange":
                typed_p = ConduitFlangeParams(**(params if isinstance(params, dict) else params.__dict__))
                generate_csharp_source(typed_p, str(cs_path))
            elif template_name == "imported_mesh":
                typed_p = ImportedMeshParams(**(params if isinstance(params, dict) else params.__dict__))
                generate_csharp_source(typed_p, str(cs_path))
            elif template_name == "ground_structure":
                from PicoGK_Dashboard.core.models import GroundStructureTrussParams
                typed_p = GroundStructureTrussParams(**(params if isinstance(params, dict) else params.__dict__))
                generate_csharp_source(typed_p, str(cs_path))
    except Exception:
        # Non-fatal if standalone code generation fails; engine will still execute
        pass

    # 4. Locate .NET CLI
    dotnet_exe = find_dotnet_executable()
    if not dotnet_exe:
        return GenerationResult(
            success=False,
            cs_path=str(cs_path) if cs_path.exists() else None,
            execution_time_sec=time.time() - start_time,
            error_message="Could not find .NET runtime / SDK on system. Please ensure .NET 8.0/9.0 is installed.",
            stderr="[PICOGK_ERROR] DotNetNotFoundException: .NET executable not found in PATH or ~/.dotnet"
        )

    csproj_path = get_csharp_project_path()
    if not csproj_path.exists():
        return GenerationResult(
            success=False,
            cs_path=str(cs_path) if cs_path.exists() else None,
            execution_time_sec=time.time() - start_time,
            error_message=f"PicoGK C# project file not found at: {csproj_path}",
            stderr=f"[PICOGK_ERROR] FileNotFoundException: {csproj_path}"
        )

    # 5. Prepare Subprocess Command
    mesh_output_base = str(out_dir_p / mesh_basename)
    expected_stl = out_dir_p / f"{mesh_basename}.stl"
    expected_obj = out_dir_p / f"{mesh_basename}.obj"

    cmd = [
        dotnet_exe,
        "run",
        "--project",
        str(csproj_path),
        "--",
        "--template",
        template_name,
        "--params",
        str(params_json_path),
        "--output-mesh",
        mesh_output_base,
        "--output-format",
        output_format,
        "--voxel-size",
        str(voxel_size_mm)
    ]

    env = get_dotnet_environment()

    # 6. Execute Subprocess
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=str(csproj_path.parent)
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return GenerationResult(
            success=False,
            cs_path=str(cs_path) if cs_path.exists() else None,
            execution_time_sec=elapsed,
            error_message=f"PicoGK generation timed out after {timeout_sec}s. Try increasing voxel size or decreasing component dimensions.",
            stderr=f"[PICOGK_ERROR] TimeoutException: Subprocess execution exceeded {timeout_sec}s threshold."
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return GenerationResult(
            success=False,
            cs_path=str(cs_path) if cs_path.exists() else None,
            execution_time_sec=elapsed,
            error_message=f"Subprocess invocation error: {str(e)}",
            stderr=traceback.format_exc()
        )

    elapsed = time.time() - start_time
    stdout_str = proc.stdout or ""
    stderr_str = proc.stderr or ""

    # 7. Parse Exit Status & Output Files
    if proc.returncode != 0:
        # Extract clean error message from stderr/stdout
        clean_error = None
        for line in (stderr_str + "\n" + stdout_str).splitlines():
            if "[PICOGK_ERROR]" in line:
                clean_error = line.replace("[PICOGK_ERROR]", "").strip()
                break
            elif "error " in line.lower() or "exception" in line.lower():
                clean_error = line.strip()
                break

        if not clean_error:
            clean_error = f"PicoGK engine exited with code {proc.returncode}."

        return GenerationResult(
            success=False,
            cs_path=str(cs_path) if cs_path.exists() else None,
            execution_time_sec=elapsed,
            stdout=stdout_str,
            stderr=stderr_str,
            error_message=clean_error
        )

    # Success: Parse JSON summary from stdout if available
    vertex_count = 0
    triangle_count = 0
    output_files: List[str] = []

    for line in stdout_str.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                summary = json.loads(line)
                if summary.get("status") == "ok":
                    vertex_count = summary.get("vertices", 0)
                    triangle_count = summary.get("triangles", 0)
                    output_files = summary.get("output_files", [])
            except Exception:
                pass

    # Verify generated files on disk
    stl_path_res = str(expected_stl) if expected_stl.exists() and expected_stl.stat().st_size > 0 else None
    obj_path_res = str(expected_obj) if expected_obj.exists() and expected_obj.stat().st_size > 0 else None

    # If output_files reported by engine have full paths, cross-check them
    for f in output_files:
        if f.lower().endswith(".stl") and os.path.exists(f) and os.path.getsize(f) > 0:
            stl_path_res = f
        elif f.lower().endswith(".obj") and os.path.exists(f) and os.path.getsize(f) > 0:
            obj_path_res = f

    return GenerationResult(
        success=True,
        stl_path=stl_path_res,
        obj_path=obj_path_res,
        cs_path=str(cs_path) if cs_path.exists() else None,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        execution_time_sec=elapsed,
        stdout=stdout_str,
        stderr=stderr_str,
        output_files=output_files or [p for p in [stl_path_res, obj_path_res] if p]
    )
