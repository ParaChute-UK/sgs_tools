import shutil
import subprocess
from pathlib import Path


def install_doc():
    """install dependencies for the documentations"""
    use_poetry = shutil.which("poetry") is not None
    print("Installating documentation dependencies...")
    if use_poetry:
        subprocess.run(["poetry", "install", "--extras", "doc"], check=True)
    else:
        print("Poetry not found. Falling back to pip...")
        subprocess.run(["pip", "install", ".[doc]"], check=True)


def build_doc():
    """compile the documentations"""

    print("Generating documentation...")
    doc_path = Path("documentation")
    subprocess.run(["sphinx-build", "-b", "html", "doc", doc_path], check=True)
    print(f"Documentation generated at: {(doc_path / 'index.html').resolve()}")


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "--setup":
        install_doc()

    build_doc()
