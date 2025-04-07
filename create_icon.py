#!/usr/bin/env python3
"""
Create a simple icon for AISearch app
"""

from PIL import Image, ImageDraw, ImageFont
import os
import sys

def create_icon():
    # Create a new image with blue background
    size = 1024
    img = Image.new('RGBA', (size, size), (42, 130, 218, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a magnifying glass
    # White circle for the lens
    circle_center = (size * 0.4, size * 0.4)
    circle_radius = size * 0.25
    draw.ellipse(
        (
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius
        ),
        outline=(255, 255, 255, 255),
        width=int(size * 0.06)
    )
    
    # White handle for the magnifying glass
    handle_width = int(size * 0.06)
    x1 = circle_center[0] + circle_radius * 0.7
    y1 = circle_center[1] + circle_radius * 0.7
    x2 = size * 0.75
    y2 = size * 0.75
    
    # Draw the handle
    draw.line((x1, y1, x2, y2), fill=(255, 255, 255, 255), width=handle_width)
    
    # Add "AI" text
    try:
        # Try to find a font
        font_size = int(size * 0.12)
        try:
            font = ImageFont.truetype("Arial Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                try:
                    # macOS system font
                    font = ImageFont.truetype("/System/Library/Fonts/SFCompact-Bold.otf", font_size)
                except:
                    # Fallback to default
                    font = ImageFont.load_default()
        
        # Draw the text inside the circle
        text = "AI"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_position = (
            circle_center[0] - text_width // 2,
            circle_center[1] - text_height // 2
        )
        
        draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)
    except Exception as e:
        print(f"Could not add text to icon: {e}")
    
    # Save the icon
    img.save("aisearch_icon.png")
    
    # Create macOS icns format icon
    try:
        # Create icns directory
        os.makedirs("AISearch.iconset", exist_ok=True)
        
        # Get the appropriate resampling filter based on PIL version
        try:
            # For Pillow 9.0.0 and above
            resampling_filter = Image.Resampling.LANCZOS
        except AttributeError:
            # For older Pillow versions
            resampling_filter = Image.Resampling.LANCZOS
        
        # Generate different sizes
        for size_name, size_value in [
            ('16x16', 16),
            ('32x32', 32),
            ('64x64', 64),
            ('128x128', 128),
            ('256x256', 256),
            ('512x512', 512),
            ('1024x1024', 1024)
        ]:
            img_resized = img.resize((size_value, size_value), resampling_filter)
            img_resized.save(f"AISearch.iconset/icon_{size_name}.png")
            
            # Save 2x version
            if size_value < 512:
                double_size = size_value * 2
                img_resized = img.resize((double_size, double_size), resampling_filter)
                img_resized.save(f"AISearch.iconset/icon_{size_name}@2x.png")
        
        # Run iconutil to create icns file (macOS only)
        if os.path.exists("/usr/bin/iconutil"):
            os.system("iconutil -c icns AISearch.iconset")
            print("Created AISearch.icns")
        else:
            print("iconutil not found. Icon set created but not converted to .icns")
    except Exception as e:
        print(f"Error creating iconset: {e}")
    
    print(f"Icon created as aisearch_icon.png")

if __name__ == "__main__":
    try:
        import PIL
    except ImportError:
        print("PIL (Pillow) not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
        import PIL
    
    create_icon() 