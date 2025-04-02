#!/usr/bin/env python3
"""
AISearch Launcher - Start either the GUI or CLI version of AISearch
"""

import sys
import os
import subprocess
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import anthropic
        import openai
        import tqdm
        gui_available = False
        try:
            from PySide6.QtWidgets import QApplication
            import markdown
            gui_available = True
        except ImportError:
            pass
        return True, gui_available
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install anthropic openai tqdm")
        print("For GUI support:")
        print("  pip install pyside6 markdown")
        return False, False

def main():
    parser = argparse.ArgumentParser(description="AISearch - AI-powered code search tool")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI version")
    parser.add_argument("--cli", action="store_true", help="Launch the CLI version")
    
    # Pass remaining arguments to the CLI version
    parser.add_argument("remaining", nargs=argparse.REMAINDER, 
                      help="Additional arguments to pass to aisearch.py")
    
    args = parser.parse_args()
    
    # Check dependencies
    deps_installed, gui_available = check_dependencies()
    if not deps_installed:
        return 1
    
    # If no specific mode is selected, use GUI if available, otherwise CLI
    if not args.gui and not args.cli:
        if gui_available:
            args.gui = True
        else:
            args.cli = True
    
    # Launch the appropriate version
    if args.gui:
        if not gui_available:
            print("GUI dependencies not installed. Please install PySide6:")
            print("  pip install pyside6")
            return 1
        
        print("Launching AISearch GUI...")
        return subprocess.call([sys.executable, "aisearch_gui.py"])
    else:
        if not args.remaining:
            # If no arguments provided, show help
            print("Launching AISearch CLI help...")
            return subprocess.call([sys.executable, "aisearch.py", "--help"])
        else:
            print("Launching AISearch CLI...")
            return subprocess.call([sys.executable, "aisearch.py"] + args.remaining)

if __name__ == "__main__":
    sys.exit(main()) 