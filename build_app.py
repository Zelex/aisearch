#!/usr/bin/env python3
"""
Build script to create a standalone AISearch app package
"""

import os
import sys
import subprocess
import shutil
import platform
import venv
import site

def create_virtual_env():
    """Create a virtual environment for building"""
    print("Creating a fresh virtual environment for building...")
    
    # Remove existing venv if it exists
    if os.path.exists("build_venv"):
        shutil.rmtree("build_venv")
    
    # Create a new virtual environment
    venv.create("build_venv", with_pip=True)
    
    # Get the path to the Python executable in the virtual environment
    if platform.system() == "Windows":
        venv_python = os.path.join("build_venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join("build_venv", "bin", "python")
    
    # Install required packages in the virtual environment
    packages = [
        "pillow",
        "pyinstaller",
        "pyside6",
        "anthropic",
        "openai",
        "tqdm",
        "markdown",
        "pygments",
    ]
    
    # Try to install pyre2, but continue if it fails
    try:
        subprocess.check_call([venv_python, "-m", "pip", "install", "pyre2"])
    except:
        print("⚠ Could not install pyre2. Continuing without it.")
    
    for package in packages:
        print(f"Installing {package} in virtual environment...")
        subprocess.check_call([venv_python, "-m", "pip", "install", package])
    
    print("✓ Virtual environment created with all dependencies.")
    return venv_python

def create_icon(python_path):
    """Create app icon"""
    print("Creating app icon...")
    if os.path.exists("create_icon.py"):
        # Update the icon creator to use Resampling.LANCZOS instead of Image.LANCZOS
        with open("create_icon.py", "r") as f:
            content = f.read()
        
        # Fix for Pillow 10+ compatibility
        if "Image.LANCZOS" in content:
            content = content.replace("Image.LANCZOS", "Image.Resampling.LANCZOS")
            with open("create_icon.py", "w") as f:
                f.write(content)
            print("✓ Updated icon script for Pillow compatibility.")
        
        # Run the icon creator
        subprocess.check_call([python_path, "create_icon.py"])
        print("✓ Icon created.")
        return True
    else:
        print("⚠ Icon creator script not found.")
        return False

def update_spec_file(icon_path):
    """Update the spec file with icon path"""
    if not os.path.exists("aisearch.spec"):
        print("⚠ aisearch.spec file not found.")
        return
    
    with open("aisearch.spec", "r") as f:
        content = f.read()
    
    # Replace icon=None with the actual icon path
    content = content.replace('icon=None', f'icon="{icon_path}"')
    
    with open("aisearch.spec", "w") as f:
        f.write(content)
    
    print("✓ Updated spec file with icon path.")

def build_app(python_path):
    """Build the standalone app"""
    print("Building AISearch app...")
    
    # Run PyInstaller
    try:
        subprocess.check_call([
            python_path, "-m", "PyInstaller", 
            "--clean",
            "aisearch.spec"
        ])
        print("✓ App built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Error building app: {e}")
        return False

def cleanup():
    """Clean up temporary build files"""
    print("Cleaning up...")
    
    # Directories to remove
    dirs_to_remove = ["build", "AISearch.iconset"]
    
    for directory in dirs_to_remove:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    
    print("✓ Cleanup complete.")

def main():
    print("====================================")
    print("AISearch App Builder")
    print("====================================")
    
    # Check platform
    system = platform.system()
    if system not in ["Darwin", "Windows", "Linux"]:
        print(f"⚠ Unsupported platform: {system}")
        return 1
    
    print(f"Building for: {system}")
    
    # Create a clean virtual environment for building
    venv_python = create_virtual_env()
    
    # Create icon
    icon_created = create_icon(venv_python)
    
    # Update spec file with icon path
    if icon_created:
        if system == "Darwin":
            icon_path = "AISearch.icns" if os.path.exists("AISearch.icns") else "aisearch_icon.png"
        elif system == "Windows":
            icon_path = "aisearch_icon.ico" if os.path.exists("aisearch_icon.ico") else "aisearch_icon.png"
        else:
            icon_path = "aisearch_icon.png"
        
        update_spec_file(icon_path)
    
    # Build the app
    if build_app(venv_python):
        # Cleanup
        cleanup()
        
        # Clean up virtual environment
        if os.path.exists("build_venv"):
            print("Removing build virtual environment...")
            shutil.rmtree("build_venv")
        
        # Print completion message
        if system == "Darwin":
            print("\nAISearch.app is now available in the dist directory!")
            print("You can copy it to your Applications folder.")
        elif system == "Windows":
            print("\nAISearch.exe is now available in the dist\\AISearch directory!")
        else:
            print("\nAISearch executable is now available in the dist/AISearch directory!")
        
        return 0
    else:
        print("\nBuild failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 