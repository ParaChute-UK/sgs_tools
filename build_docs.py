import shutil
import subprocess
from pathlib import Path


def build_docs():
    """script to compile the documentations"""
    use_poetry = shutil.which("poetry") is not None
    print("Generating documentation...")
    doc_path = Path("documentation")

    if use_poetry:
        subprocess.run(["poetry", "install", "--with", "doc"])
        subprocess.run(["poetry", "run", "sphinx-build", "-b", "html", "doc", doc_path])
    else:
        print("Poetry not found. Falling back to pip...")
        subprocess.run(["pip", "install", ".[doc]"])
        subprocess.run(["sphinx-build", "-b", "html", "doc", doc_path])

    out = doc_path / "index.html"
    print(f"Documentation generated at: {out.resolve()}")
