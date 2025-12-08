#!/usr/bin/env python3
"""
Automatic Setup & Launcher for Product Sales Data Mining Project
Creates venv, installs deps, builds notebook, launches Jupyter automatically.
"""

import os
import sys
import platform
import subprocess

PROJECT_NOTEBOOK = "ML_Analysis.ipynb"
REQUIREMENTS_FILE = "requirements.txt"
VENV_DIR = "venv"


# --------------------- Helpers ---------------------

def run(cmd, exit_on_fail=True):
    """Run shell commands with pretty printing."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and exit_on_fail:
        print("❌ Command failed.")
        sys.exit(1)
    return result.returncode == 0


def ensure_requirements_file():
    """Create requirements.txt if missing or empty."""
    if not os.path.exists(REQUIREMENTS_FILE) or os.path.getsize(REQUIREMENTS_FILE) == 0:
        print("📄 Creating default requirements.txt...")
        with open(REQUIREMENTS_FILE, "w") as f:
            f.write(
                "numpy\n"
                "pandas\n"
                "matplotlib\n"
                "seaborn\n"
                "scikit-learn\n"
                "scipy\n"
                "jupyter\n"
            )
        print("✓ requirements.txt created.")
    else:
        print("✓ requirements.txt already exists.")


def create_venv():
    """Create virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print("📦 Creating virtual environment...")
        run(f"python3 -m venv {VENV_DIR}")
        print("✓ Virtual environment created.")
    else:
        print("✓ venv already exists.")


def get_venv_python():
    """Return path to Python inside venv."""
    system = platform.system().lower()
    if system == "windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python3")


def install_requirements(venv_python):
    """Install dependencies from requirements.txt."""
    print("\n📥 Installing dependencies...")
    run(f"{venv_python} -m pip install --upgrade pip", exit_on_fail=True)
    run(f"{venv_python} -m pip install -r {REQUIREMENTS_FILE}", exit_on_fail=True)
    print("✓ All dependencies installed!")


def ensure_notebook_exists(venv_python):
    """Run create_notebook.py if the notebook is missing."""
    if not os.path.exists(PROJECT_NOTEBOOK):
        print("🧾 Notebook missing — generating with create_notebook.py...")
        if os.path.exists("create_notebook.py"):
            run(f"{venv_python} create_notebook.py")
            print(f"✓ {PROJECT_NOTEBOOK} generated!")
        else:
            print("❌ create_notebook.py not found. Cannot auto-generate notebook.")
            sys.exit(1)
    else:
        print(f"✓ {PROJECT_NOTEBOOK} exists.")


def launch_jupyter(venv_python):
    """Launch the notebook in Jupyter."""
    print("\n🚀 Launching Jupyter Notebook...")
    run(f"{venv_python} -m notebook {PROJECT_NOTEBOOK}", exit_on_fail=False)


# --------------------- Main Setup Flow ---------------------

def main():
    print("=" * 70)
    print("🔧 PRODUCT SALES DATA MINING PROJECT — AUTOMATIC SETUP")
    print("=" * 70)

    # Ensure requirements exist
    ensure_requirements_file()

    # Create virtual environment
    create_venv()

    # Resolve venv python path
    venv_python = get_venv_python()

    # Install dependencies
    install_requirements(venv_python)

    # Ensure notebook exists
    ensure_notebook_exists(venv_python)

    # Launch Jupyter
    launch_jupyter(venv_python)

    print("\n✨ Setup complete! Jupyter should now be open.")
    print("If it did not open, manually run:")
    print(f"  {venv_python} -m notebook {PROJECT_NOTEBOOK}")


if __name__ == "__main__":
    main()
