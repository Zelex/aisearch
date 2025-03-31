#!/usr/bin/env python3
"""
AISearch Installer - Install dependencies for AISearch
"""

import sys
import subprocess
import os

def main():
    print("Installing AISearch dependencies...\n")
    
    # Install basic dependencies
    print("Installing core dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "openai", "tqdm"])
    
    # Ask user if they want GUI support
    install_gui = input("\nDo you want to install GUI support? (y/n): ").lower() == 'y'
    
    if install_gui:
        print("\nInstalling GUI dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyside6"])
        print("\nGUI dependencies installed!")
    
    # Ask for API keys if not set
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not anthropic_key:
        print("\nAnthropic API key not found in environment variables.")
        set_key = input("Do you want to set your Anthropic API key? (y/n): ").lower() == 'y'
        
        if set_key:
            anthropic_key = input("Enter your Anthropic API key: ")
            set_env_var("ANTHROPIC_API_KEY", anthropic_key)
    
    if not openai_key:
        print("\nOpenAI API key not found in environment variables.")
        set_key = input("Do you want to set your OpenAI API key? (y/n): ").lower() == 'y'
        
        if set_key:
            openai_key = input("Enter your OpenAI API key: ")
            set_env_var("OPENAI_API_KEY", openai_key)
    
    print("\nInstallation complete!")
    print("\nTo start AISearch:")
    print("- GUI:  ./run_aisearch.py --gui")
    print("- CLI:  ./run_aisearch.py --cli <directory> --prompt \"your search prompt\"")
    print("- Auto: ./run_aisearch.py (selects GUI if available, otherwise CLI)")
    
    return 0

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

if __name__ == "__main__":
    sys.exit(main()) 