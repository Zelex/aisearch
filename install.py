#!/usr/bin/env python3
"""
AISearch Installer - Install dependencies for AISearch
"""

import sys
import subprocess
import os
import venv
import json
import re
from pathlib import Path

# Version requirements
REQUIREMENTS = {
    "anthropic": ">=0.18.1",
    "openai": ">=1.12.0",
    "tqdm": ">=4.66.1",
    "pyside6": ">=6.6.1",
    "markdown": ">=3.4.0",
    "regex": ">=2022.1.18"
}

# Optional performance requirements
OPTIONAL_REQUIREMENTS = {
    "pygments": ">=2.16.1"  # Syntax highlighting in GUI
}

def validate_api_key(key, provider):
    """Validate API key format"""
    if provider == "anthropic":
        # Anthropic keys start with 'sk-ant-'
        return bool(re.match(r'^sk-ant-[a-zA-Z0-9]+$', key))
    elif provider == "openai":
        # OpenAI keys start with 'sk-'
        return bool(re.match(r'^sk-[a-zA-Z0-9]+$', key))
    return False

def create_requirements_file(include_optional=None):
    """Create requirements.txt with pinned versions
    
    Args:
        include_optional: List of optional packages to include
    """
    with open("requirements.txt", "w") as f:
        # Core requirements
        for package, version in REQUIREMENTS.items():
            f.write(f"{package}{version}\n")
        
        # Optional requirements if specified
        if include_optional:
            for package in include_optional:
                if package in OPTIONAL_REQUIREMENTS:
                    version = OPTIONAL_REQUIREMENTS[package]
                    f.write(f"{package}{version}\n")

def install_package(package, version):
    """Install a package with progress indicator"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            f"{package}{version}", "--no-cache-dir"
        ])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def create_virtual_env():
    """Create a virtual environment"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        try:
            venv.create("venv", with_pip=True)
            print("✓ Virtual environment created successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to create virtual environment: {e}")
            return False
    return True

def set_env_var(var_name, value):
    """Set environment variable and provide instructions for permanent setup"""
    if sys.platform == 'win32':
        # Windows
        print(f"\nTo set the API key permanently, run the following command in Command Prompt:")
        print(f"setx {var_name} {value}")
        
        # Set for current session
        os.environ[var_name] = value
    else:
        # macOS/Linux
        shell = os.environ.get("SHELL", "/bin/bash")
        
        if "bash" in shell:
            profile_file = os.path.expanduser("~/.bash_profile")
        elif "zsh" in shell:
            profile_file = os.path.expanduser("~/.zshrc")
        else:
            profile_file = os.path.expanduser("~/.profile")
        
        print(f"\nTo set the API key permanently, add this line to {profile_file}:")
        print(f"export {var_name}={value}")
        
        # Set for current session
        os.environ[var_name] = value

def cleanup_failed_install():
    """Clean up any partially installed files"""
    try:
        if os.path.exists("requirements.txt"):
            os.remove("requirements.txt")
        if os.path.exists("venv"):
            import shutil
            shutil.rmtree("venv")
    except Exception as e:
        print(f"Warning: Failed to clean up: {e}")

def main():
    print("AISearch Installer\n")
    
    # Check if running in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    # Ask if user wants to use a virtual environment
    if not in_venv:
        use_venv = input("Do you want to create a virtual environment? (y/n): ").lower() == 'y'
        if use_venv:
            if not create_virtual_env():
                cleanup_failed_install()
                return 1
            # Activate virtual environment
            if sys.platform == 'win32':
                activate_script = os.path.join("venv", "Scripts", "activate.bat")
            else:
                activate_script = os.path.join("venv", "bin", "activate")
            print(f"\nPlease activate the virtual environment:")
            print(f"  {sys.platform}: {activate_script}")
            print("Then run this script again.")
            return 0
    
    # Track optional packages to install
    optional_packages = []
    
    # Ask user if they want GUI support
    install_gui = input("\nDo you want to install GUI support? (y/n): ").lower() == 'y'
    
    # Ask user if they want syntax highlighting for GUI
    install_pygments = False
    if install_gui:
        install_pygments = input("\nDo you want to install syntax highlighting for the GUI? (y/n): ").lower() == 'y'
        if install_pygments:
            optional_packages.append("pygments")
    
    # Create requirements.txt with selected optional packages
    create_requirements_file(include_optional=optional_packages)
    
    # Install core dependencies
    print("\nInstalling core dependencies...")
    success = True
    for package, version in REQUIREMENTS.items():
        if package != "pyside6":  # GUI dependency will be handled separately
            if not install_package(package, version):
                if package == "regex":
                    print("Warning: Failed to install regex. Falling back to standard re module.")
                    print("Note: Search performance may be slower with complex patterns.")
                else:
                    success = False
                    break
    
    if not success:
        cleanup_failed_install()
        return 1
    
    # Install GUI if requested
    if install_gui:
        print("\nInstalling GUI dependencies...")
        if not install_package("pyside6", REQUIREMENTS["pyside6"]):
            cleanup_failed_install()
            return 1
        print("\n✓ GUI dependencies installed!")
    
    # Install Pygments if requested
    if install_gui and install_pygments:
        print("\nInstalling syntax highlighting library...")
        if not install_package("pygments", OPTIONAL_REQUIREMENTS["pygments"]):
            print("Failed to install Pygments. Code highlighting will be disabled.")
        else:
            print("\n✓ Syntax highlighting library installed!")
    
    # Ask for API keys if not set
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not anthropic_key:
        print("\nAnthropic API key not found in environment variables.")
        set_key = input("Do you want to set your Anthropic API key? (y/n): ").lower() == 'y'
        
        if set_key:
            while True:
                anthropic_key = input("Enter your Anthropic API key: ")
                if validate_api_key(anthropic_key, "anthropic"):
                    set_env_var("ANTHROPIC_API_KEY", anthropic_key)
                    break
                else:
                    print("Invalid Anthropic API key format. Please try again.")
    
    if not openai_key:
        print("\nOpenAI API key not found in environment variables.")
        set_key = input("Do you want to set your OpenAI API key? (y/n): ").lower() == 'y'
        
        if set_key:
            while True:
                openai_key = input("Enter your OpenAI API key: ")
                if validate_api_key(openai_key, "openai"):
                    set_env_var("OPENAI_API_KEY", openai_key)
                    break
                else:
                    print("Invalid OpenAI API key format. Please try again.")
    
    print("\n✓ Installation complete!")
    
    print("\nTo start AISearch:")
    print("- GUI:  ./run_aisearch.py --gui")
    print("- CLI:  ./run_aisearch.py --cli <directory> --prompt \"your search prompt\"")
    print("- Auto: ./run_aisearch.py (selects GUI if available, otherwise CLI)")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user.")
        cleanup_failed_install()
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        cleanup_failed_install()
        sys.exit(1) 