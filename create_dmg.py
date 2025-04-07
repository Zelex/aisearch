#!/usr/bin/env python3
"""
Create a DMG file for AISearch app distribution
"""

import os
import sys
import subprocess
import platform

def create_dmg():
    """Create a DMG file for the AISearch app"""
    # Check if we're on macOS
    if platform.system() != "Darwin":
        print("This script is for macOS only.")
        return False
    
    # Check if app exists
    if not os.path.exists("dist/AISearch.app"):
        print("AISearch.app not found in dist directory.")
        print("Run build_app.py first to create the app.")
        return False
    
    # Create DMG
    print("Creating DMG file...")
    
    # Create temporary directory for DMG contents
    os.makedirs("dist/dmg", exist_ok=True)
    
    # Copy app to temporary directory
    subprocess.check_call(["cp", "-r", "dist/AISearch.app", "dist/dmg/"])
    
    # Create a symbolic link to /Applications
    os.chdir("dist/dmg")
    subprocess.check_call(["ln", "-s", "/Applications", "Applications"])
    os.chdir("../..")
    
    # Create DMG
    dmg_file = "AISearch.dmg"
    if os.path.exists(dmg_file):
        os.remove(dmg_file)
    
    # Create DMG file
    subprocess.check_call([
        "hdiutil", "create",
        "-volname", "AISearch",
        "-srcfolder", "dist/dmg",
        "-ov", "-format", "UDZO",
        dmg_file
    ])
    
    # Clean up
    subprocess.check_call(["rm", "-rf", "dist/dmg"])
    
    print(f"✓ DMG created: {dmg_file}")
    return True

if __name__ == "__main__":
    create_dmg() 