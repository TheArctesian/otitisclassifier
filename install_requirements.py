#!/usr/bin/env python3
"""
Install required packages for Binary Screening Model development.

This script checks for and installs the necessary packages for the 
dual-architecture medical AI system, particularly focusing on the
Binary Screening Model requirements.
"""

import subprocess
import sys
import importlib


def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Install required packages for Binary Screening Model."""
    print("Installing Binary Screening Model Requirements")
    print("=" * 50)
    
    # Critical packages for Binary Screening Model
    required_packages = {
        'torch': 'torch>=2.0.0',
        'torchvision': 'torchvision>=0.15.0',
        'timm': 'timm>=0.9.0',
        'numpy': 'numpy>=1.24.0',
        'PIL': 'Pillow>=10.0.0',
        'cv2': 'opencv-python>=4.8.0',
        'sklearn': 'scikit-learn>=1.3.0',
        'pandas': 'pandas>=2.0.0',
        'matplotlib': 'matplotlib>=3.7.0'
    }
    
    missing_packages = []
    
    # Check which packages are missing
    for import_name, pip_name in required_packages.items():
        if check_package(import_name):
            print(f"OK: {import_name} is already installed")
        else:
            print(f"MISSING: {import_name}")
            missing_packages.append(pip_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}")
                print(f"Please install manually: pip install {package}")
    
    else:
        print("\nAll required packages are already installed!")
    
    # Verify installation
    print("\nVerifying installation...")
    all_installed = True
    
    for import_name, _ in required_packages.items():
        if check_package(import_name):
            print(f"OK: {import_name}")
        else:
            print(f"MISSING: {import_name} - installation failed")
            all_installed = False
    
    if all_installed:
        print("\nBinary Screening Model environment is ready!")
        print("You can now run: python test_binary_screening.py")
    else:
        print("\nSome packages failed to install. Please install them manually.")
        print("See requirements.txt for the complete list.")


if __name__ == "__main__":
    main()