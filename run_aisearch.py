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
        import tqdm
        gui_available = False
        try:
            from PySide6.QtWidgets import QApplication
            gui_available = True
        except ImportError:
            pass
        return True, gui_available
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install anthropic tqdm")
        print("For GUI support:")
        print("  pip install pyside6")
        return False, False

def main():
    parser = argparse.ArgumentParser(description="AISearch - AI-powered code search tool")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI version")
    parser.add_argument("--cli", action="store_true", help="Launch the CLI version")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    
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
    
    # Set environment variables for debugging and fixing multiprocessing issues
    if args.debug:
        os.environ["AISEARCH_DEBUG"] = "1"
        
    # Set environment variables to fix multiprocessing issues on macOS
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    
    # Disable multiprocessing resource tracking to prevent semaphore leaks
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"
    
    # Run multiprocessing in spawn mode for better compatibility
    os.environ["PYTHONFAULTHANDLER"] = "1"  # Enable fault handler
    
    if sys.platform == 'darwin':
        print("Detected macOS platform, applying special multiprocessing fixes")
        # Apply macOS-specific fixes
        os.environ["PYDEVD_USE_FRAME_EVAL"] = "NO"  # Fix for debugger on macOS
        
    if args.debug:
        print("Debug mode enabled")
        # Print Python and system information
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        # Debug mode enables extra diagnostic printouts
        os.environ["PYTHONDEBUG"] = "1"
        
    # Launch the appropriate version
    if args.gui:
        if not gui_available:
            print("GUI dependencies not installed. Please install PySide6:")
            print("  pip install pyside6")
            return 1
        
        print("Launching AISearch GUI...")
        
        # Pass environment variables through to subprocess
        env = os.environ.copy()
        
        # Add special Qt flags to prevent timer issues
        env["QT_LOGGING_RULES"] = "qt.qpa.timer=false"
        
        if args.debug:
            return subprocess.call([sys.executable, "aisearch_gui.py"], env=env)
        else:
            return subprocess.call([sys.executable, "aisearch_gui.py"], env=env)
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