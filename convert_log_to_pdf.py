import os
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    root = Path(__file__).resolve().parent
    tex_path = root / "docs" / "TO_3D_log.tex"
    if not tex_path.exists():
        tex_path = root / "TO_3D_log.tex"
    
    if not tex_path.exists():
        print(f"Error: {tex_path} does not exist")
        return 1

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        # Fallback to local MiKTeX install
        default_miktex = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe"
        if default_miktex.exists():
            pdflatex = str(default_miktex)

    if not pdflatex:
        print("Error: pdflatex not found. Please ensure MiKTeX or TeX Live is installed.")
        return 1

    print(f"Compiling {tex_path.name} with {pdflatex}...")
    out_dir = tex_path.parent
    cmd = [pdflatex, "-interaction=batchmode", f"-output-directory={out_dir}", str(tex_path)]
    
    # 2 passes for cross-references
    subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True)

    pdf_out = out_dir / "TO_3D_log.pdf"
    if pdf_out.exists():
        # Copy to root as well
        shutil.copy2(pdf_out, root / "TO_3D_log.pdf")
        print(f"[SUCCESS] Compiled publication-grade PDF: {pdf_out} ({pdf_out.stat().st_size:,} bytes)")
        return 0
    else:
        print("Error: PDF output was not generated.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
