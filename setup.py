#!/usr/bin/env python3
"""
Setup and Execution Script for Product Sales Data Mining Project
Verifies dependencies and prepares environment for analysis
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✓ Python {sys.version.split()[0]} (OK)")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False
        return True
    except (ImportError, ModuleNotFoundError):
        return False

def main():
    print("=" * 70)
    print("PRODUCT SALES DATA MINING PROJECT - SETUP VERIFICATION")
    print("=" * 70)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        print("   Please upgrade to Python 3.8 or higher")
        return False
    
    # Check required packages
    print("\n2. Checking required packages...")
    
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("jupyter", "jupyter"),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            print(f"   ✓ {package_name}")
        else:
            print(f"   ❌ {package_name} (MISSING)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n3. Installing missing packages...")
        packages_str = " ".join(missing_packages)
        cmd = f"pip install {packages_str}"
        print(f"   Running: {cmd}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("   ✓ Packages installed successfully")
        except subprocess.CalledProcessError:
            print("   ❌ Installation failed")
            return False
    else:
        print("\n3. All packages already installed ✓")
    
    # Check data file
    print("\n4. Checking data file...")
    import os
    if os.path.exists("product_sales.csv"):
        print("   ✓ product_sales.csv found")
    else:
        print("   ❌ product_sales.csv not found")
        print("   Please ensure product_sales.csv is in the project directory")
        return False
    
    # Check notebook file
    print("\n5. Checking notebook file...")
    if os.path.exists("ML_Analysis.ipynb"):
        print("   ✓ ML_Analysis.ipynb found")
    else:
        print("   ❌ ML_Analysis.ipynb not found")
        print("   Please ensure ML_Analysis.ipynb is in the project directory")
        return False
    
    print("\n" + "=" * 70)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 70)
    print("\n✓ All systems ready!")
    print("\nTo run the analysis:")
    print("  1. Open terminal/command prompt in this directory")
    print("  2. Run: jupyter notebook ML_Analysis.ipynb")
    print("  3. Run all cells (Ctrl+A, then Ctrl+Enter)")
    print("\nEstimated runtime: 5-10 minutes")
    print("\nProject Structure:")
    print("  ├── ML_Analysis.ipynb       (Main analysis - RUN THIS)")
    print("  ├── product_sales.csv       (Dataset)")
    print("  ├── preprocessing.py        (Data preprocessing)")
    print("  ├── kmeans.py               (K-means implementation)")
    print("  ├── regression.py           (Regression models)")
    print("  ├── visualization.py        (Visualization utilities)")
    print("  ├── README.md               (Project overview)")
    print("  └── PROJECT_GUIDE.md        (Comprehensive guide)")
    print("\n" + "=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
