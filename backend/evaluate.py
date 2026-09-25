import sys
from pathlib import Path

# Add the backend directory and src directory to Python path
backend_dir = Path(__file__).parent.resolve()
src_dir = backend_dir / "src"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(src_dir))

from evaluation.cli import run_cli

if __name__ == "__main__":
    run_cli()
