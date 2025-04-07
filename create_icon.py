#!/usr/bin/env python3
"""
Create a premium modern icon for AISearch app with advanced visual styling
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance, ImageChops
import os
import sys
import math
import random
from io import BytesIO

def create_radial_gradient(width, height, center_color, outer_color):
    """Create a radial gradient"""
    image = Image.new('RGBA', (width, height))
    
    # Calculate maximum distance from center (corner distance)
    center = (width // 2, height // 2)
    max_distance = math.sqrt(center[0]**2 + center[1]**2)
    
    for y in range(height):
        for x in range(width):
            # Calculate distance to center
            distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            # Normalize distance (0-1)
            normalized_distance = distance / max_distance
            # Calculate color interpolation
            r = int(center_color[0] * (1 - normalized_distance) + outer_color[0] * normalized_distance)
            g = int(center_color[1] * (1 - normalized_distance) + outer_color[1] * normalized_distance)
            b = int(center_color[2] * (1 - normalized_distance) + outer_color[2] * normalized_distance)
            a = int(center_color[3] * (1 - normalized_distance) + outer_color[3] * normalized_distance)
            
            image.putpixel((x, y), (r, g, b, a))
            
    return image

def create_multi_gradient(width, height, colors, direction='vertical'):
    """Create a multi-color gradient"""
    if len(colors) < 2:
        raise ValueError("At least 2 colors are needed for a gradient")
    
    base = Image.new('RGBA', (width, height), colors[0])
    
    # Calculate segment height/width
    if direction == 'vertical':
        segment_size = height / (len(colors) - 1)
    else:  # horizontal
        segment_size = width / (len(colors) - 1)
    
    # For each pair of adjacent colors, create a gradient
    for i in range(len(colors) - 1):
        if direction == 'vertical':
            # Vertical slice
            top = i * segment_size
            bottom = (i + 1) * segment_size
            grad_height = int(bottom - top)
            gradient = Image.new('RGBA', (width, grad_height))
            
            for y in range(grad_height):
                # Calculate interpolation factor
                factor = y / grad_height
                r = int(colors[i][0] * (1 - factor) + colors[i+1][0] * factor)
                g = int(colors[i][1] * (1 - factor) + colors[i+1][1] * factor)
                b = int(colors[i][2] * (1 - factor) + colors[i+1][2] * factor)
                a = int(colors[i][3] * (1 - factor) + colors[i+1][3] * factor)
                
                # Create a horizontal line with this color
                for x in range(width):
                    gradient.putpixel((x, y), (r, g, b, a))
            
            # Paste this gradient segment onto base
            base.paste(gradient, (0, int(top)))
        else:  # horizontal
            # Horizontal slice
            left = i * segment_size
            right = (i + 1) * segment_size
            grad_width = int(right - left)
            gradient = Image.new('RGBA', (grad_width, height))
            
            for x in range(grad_width):
                # Calculate interpolation factor
                factor = x / grad_width
                r = int(colors[i][0] * (1 - factor) + colors[i+1][0] * factor)
                g = int(colors[i][1] * (1 - factor) + colors[i+1][1] * factor)
                b = int(colors[i][2] * (1 - factor) + colors[i+1][2] * factor)
                a = int(colors[i][3] * (1 - factor) + colors[i+1][3] * factor)
                
                # Create a vertical line with this color
                for y in range(height):
                    gradient.putpixel((x, y), (r, g, b, a))
            
            # Paste this gradient segment onto base
            base.paste(gradient, (int(left), 0))
    
    return base

def create_noise_texture(width, height, opacity=10):
    """Create subtle noise texture for depth"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    pixels = img.load()
    
    for i in range(width):
        for j in range(height):
            noise_val = random.randint(0, 255)
            pixels[i, j] = (noise_val, noise_val, noise_val, opacity)
    
    return img

def add_rounded_corners(image, radius):
    """Add rounded corners to an image"""
    # Create a mask for rounded corners
    circle = Image.new('L', (radius * 2, radius * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)
    
    width, height = image.size
    mask = Image.new('L', (width, height), 255)
    
    # Paste corner circles
    mask.paste(circle.crop((0, 0, radius, radius)), (0, 0))
    mask.paste(circle.crop((radius, 0, radius * 2, radius)), (width - radius, 0))
    mask.paste(circle.crop((0, radius, radius, radius * 2)), (0, height - radius))
    mask.paste(circle.crop((radius, radius, radius * 2, radius * 2)), (width - radius, height - radius))
    
    # Apply the mask to the image
    result = image.copy()
    result.putalpha(mask)
    
    return result

def create_glass_effect(size, center, radius, color=(255, 255, 255, 180)):
    """Create a more realistic glass effect with reflections"""
    glass = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glass)
    
    # Base glass circle with alpha
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius
        ),
        fill=color
    )
    
    # Add highlight reflections (curved shapes)
    highlight_offset = radius * 0.2
    highlight_size = radius * 0.6
    highlight_pos = (center[0] - highlight_offset, center[1] - highlight_offset)
    
    # Top-left highlight (brighter)
    draw.ellipse(
        (
            highlight_pos[0] - highlight_size,
            highlight_pos[1] - highlight_size,
            highlight_pos[0] + highlight_size,
            highlight_pos[1] + highlight_size
        ),
        fill=(255, 255, 255, 90)
    )
    
    # Bottom-right shadow
    shadow_offset = radius * 0.1
    shadow_pos = (center[0] + shadow_offset, center[1] + shadow_offset)
    
    draw.ellipse(
        (
            shadow_pos[0] - radius * 0.9,
            shadow_pos[1] - radius * 0.9,
            shadow_pos[0] + radius * 0.9,
            shadow_pos[1] + radius * 0.9
        ),
        fill=(0, 0, 0, 15)
    )
    
    # Apply subtle blur for realistic glass
    glass = glass.filter(ImageFilter.GaussianBlur(radius=3))
    
    return glass

def add_shadow(image, offset=(30, 30), shadow_color=(0, 0, 0, 60), blur_radius=15):
    """Add a drop shadow to an image"""
    # Make a copy of the image to become the shadow
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow.paste(shadow_color, (0, 0, image.size[0], image.size[1]), image)
    
    # Blur the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Create a new image with space for the shadow
    result = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Paste the shadow first, offset from the image
    result.paste(shadow, offset, shadow)
    
    # Paste the original image on top
    result.paste(image, (0, 0), image)
    
    return result

def create_metal_rim(size, center, radius, rim_width):
    """Create a realistic metallic rim"""
    rim = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rim)
    
    # Create base rim
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius
        ),
        outline=(200, 200, 200, 255),
        width=rim_width
    )
    
    # Apply a gradient to make it look metallic
    # We'll create a separate image with a gradient and use it as a mask
    gradient = Image.new('L', (size, size), 0)
    grad_draw = ImageDraw.Draw(gradient)
    
    # Draw a gradient from top-left to bottom-right
    for i in range(0, 360, 5):  # Step every 5 degrees
        angle_rad = math.radians(i)
        x1 = center[0] + (radius - rim_width/2) * math.cos(angle_rad)
        y1 = center[1] + (radius - rim_width/2) * math.sin(angle_rad)
        
        # Brightness varies with angle
        brightness = int(127 + 127 * math.sin(angle_rad * 2))
        
        # Draw a small circle at this point with the calculated brightness
        grad_draw.ellipse(
            (x1-2, y1-2, x1+2, y1+2),
            fill=brightness
        )
    
    # Blur the gradient for smooth transitions
    gradient = gradient.filter(ImageFilter.GaussianBlur(rim_width/2))
    
    # Create metallic rim image
    metal_color1 = (220, 220, 220, 255)  # Light silver
    metal_color2 = (120, 120, 120, 255)  # Dark silver
    
    metal_rim = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    for y in range(size):
        for x in range(size):
            # Only process pixels inside the rim
            dx = x - center[0]
            dy = y - center[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if abs(distance - radius) <= rim_width/2:
                # Get gradient value for this point
                gradient_value = gradient.getpixel((x, y)) / 255.0
                
                # Interpolate between two metal colors
                r = int(metal_color1[0] * gradient_value + metal_color2[0] * (1 - gradient_value))
                g = int(metal_color1[1] * gradient_value + metal_color2[1] * (1 - gradient_value))
                b = int(metal_color1[2] * gradient_value + metal_color2[2] * (1 - gradient_value))
                
                metal_rim.putpixel((x, y), (r, g, b, 255))
    
    return metal_rim

def create_icon():
    # Create a new image with multi-color gradient background
    size = 1024
    
    # Modern blue to purple gradient colors
    gradient_colors = [
        (88, 101, 242, 255),    # Discord blue
        (64, 93, 230, 255),     # Rich blue
        (59, 130, 246, 255),    # Bright blue
        (79, 70, 229, 255),     # Indigo
        (124, 58, 237, 255),    # Purple
    ]
    
    # Create beautiful diagonal gradient
    img = create_multi_gradient(size, size, gradient_colors, direction='vertical')
    draw = ImageDraw.Draw(img)
    
    # Add subtle noise texture for depth
    noise = create_noise_texture(size, size, opacity=5)
    img.paste(noise, (0, 0), noise)
    
    # Draw a magnifying glass with modern styling
    circle_center = (int(size * 0.4), int(size * 0.4))
    circle_radius = int(size * 0.25)
    
    # Create realistic glass effect
    glass = create_glass_effect(size, circle_center, circle_radius, color=(255, 255, 255, 130))
    
    # Create metallic rim for the magnifying glass
    rim_width = int(size * 0.05)
    metal_rim = create_metal_rim(size, circle_center, circle_radius, rim_width)
    
    # White handle for the magnifying glass with 3D effect
    handle_width = int(size * 0.09)
    handle_length = int(size * 0.3)
    
    # Calculate handle coordinates
    angle = math.radians(45)  # 45 degree angle
    x1 = circle_center[0] + (circle_radius - rim_width/2) * math.cos(angle)
    y1 = circle_center[1] + (circle_radius - rim_width/2) * math.sin(angle)
    x2 = x1 + handle_length * math.cos(angle)
    y2 = y1 + handle_length * math.sin(angle)
    
    # Create handle shape mask
    handle_mask = Image.new('L', (size, size), 0)
    handle_draw = ImageDraw.Draw(handle_mask)
    
    # Draw thick line for handle
    for i in range(-handle_width//2, handle_width//2):
        # Calculate offset perpendicular to the line
        perpendicular_angle = angle + math.pi/2
        offset_x = i * math.cos(perpendicular_angle)
        offset_y = i * math.sin(perpendicular_angle)
        
        handle_draw.line(
            (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), 
            fill=255, 
            width=1
        )
    
    # Draw end cap
    end_cap_radius = handle_width / 2
    handle_draw.ellipse(
        (
            x2 - end_cap_radius,
            y2 - end_cap_radius,
            x2 + end_cap_radius,
            y2 + end_cap_radius
        ),
        fill=255
    )
    
    # Blur the mask slightly for smoother edges
    handle_mask = handle_mask.filter(ImageFilter.GaussianBlur(1))
    
    # Create metallic handle gradient
    handle_gradient = create_multi_gradient(size, size, [
        (240, 240, 240, 255),  # Light silver
        (180, 180, 180, 255),  # Medium silver 
        (220, 220, 220, 255),  # Light silver
    ], direction='horizontal')
    
    # Apply the mask to handle gradient
    handle_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    handle_img.paste(handle_gradient, mask=handle_mask)
    
    # Apply drop shadow to handle
    handle_shadow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(handle_shadow)
    
    shadow_offset = 3  # shadow offset in pixels
    for i in range(-handle_width//2, handle_width//2):
        perpendicular_angle = angle + math.pi/2
        offset_x = i * math.cos(perpendicular_angle)
        offset_y = i * math.sin(perpendicular_angle)
        
        shadow_draw.line(
            (
                x1 + offset_x + shadow_offset, 
                y1 + offset_y + shadow_offset, 
                x2 + offset_x + shadow_offset, 
                y2 + offset_y + shadow_offset
            ), 
            fill=(0, 0, 0, 40), 
            width=1
        )
    
    shadow_draw.ellipse(
        (
            x2 - end_cap_radius + shadow_offset,
            y2 - end_cap_radius + shadow_offset,
            x2 + end_cap_radius + shadow_offset,
            y2 + end_cap_radius + shadow_offset
        ),
        fill=(0, 0, 0, 40)
    )
    
    shadow_handle = shadow_handle = handle_shadow.filter(ImageFilter.GaussianBlur(4))
    
    # Paste all elements in correct order
    img.paste(shadow_handle, (0, 0), shadow_handle)  # Handle shadow first
    img.paste(handle_img, (0, 0), handle_img)        # Then handle
    img.paste(glass, (0, 0), glass)                  # Glass effect in lens
    img.paste(metal_rim, (0, 0), metal_rim)          # Metal rim last
    
    # Add "AI" text with premium styling
    try:
        # Try to find a modern tech font
        font_size = int(size * 0.14)
        try:
            # Try different modern fonts in order
            font_candidates = [
                "SF-Pro-Display-Bold.otf",
                "SFPro-Bold.ttf",
                "/System/Library/Fonts/SFCompact-Bold.otf",
                "/System/Library/Fonts/SFPro-Bold.otf",
                "/Library/Fonts/SF-Pro-Display-Bold.otf",
                "Arial-Bold.ttf",
                "Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            ]
            
            font = None
            for font_path in font_candidates:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Draw the text inside the circle
        text = "AI"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position text in the center of the lens
        text_position = (
            circle_center[0] - text_width // 2,
            circle_center[1] - text_height // 2
        )
        
        # Create text with a modern 3D effect
        # Create a text layer
        text_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        
        # Shadow layers (multiple for deeper effect)
        for offset in range(1, 8, 2):
            alpha = 60 - offset * 7  # Reduce alpha as offset increases
            if alpha < 0:
                alpha = 0
                
            text_draw.text(
                (text_position[0] + offset, text_position[1] + offset), 
                text, 
                fill=(0, 0, 0, alpha), 
                font=font
            )
        
        # Main text with gradient effect
        text_gradient = create_multi_gradient(text_width, text_height, [
            (255, 255, 255, 255),  # White
            (240, 240, 240, 255),  # Light gray
            (255, 255, 255, 255),  # White
        ], direction='vertical')
        
        # Create a mask with the text
        text_mask = Image.new('L', (size, size), 0)
        text_mask_draw = ImageDraw.Draw(text_mask)
        text_mask_draw.text(text_position, text, fill=255, font=font)
        
        # Apply text mask to gradient
        text_gradient_resized = text_gradient.resize((text_width, text_height))
        text_colored = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        text_colored.paste(text_gradient_resized, (text_position[0], text_position[1]))
        
        # Apply the text mask
        text_final = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        text_final.paste(text_colored, mask=text_mask)
        
        # Add a subtle glow around text
        text_glow = text_mask.filter(ImageFilter.GaussianBlur(5))
        text_glow_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        
        # Use the blurred mask to create a glow
        for x in range(size):
            for y in range(size):
                alpha = text_glow.getpixel((x, y))
                if alpha > 0:
                    text_glow_layer.putpixel((x, y), (255, 255, 255, min(alpha, 40)))
        
        # Paste text shadow, glow, and main text
        img.paste(text_layer, (0, 0), text_layer)  # Shadow first
        img.paste(text_glow_layer, (0, 0), text_glow_layer)  # Glow next
        img.paste(text_final, (0, 0), text_final)  # Main text on top
    
    except Exception as e:
        print(f"Could not add text to icon: {e}")
    
    # Apply depth effects
    
    # Add a subtle inner shadow inside the icon edges
    inner_shadow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(inner_shadow)
    
    # Draw a slightly smaller rectangle with transparency
    shadow_margin = 20
    shadow_draw.rectangle(
        (shadow_margin, shadow_margin, size - shadow_margin, size - shadow_margin),
        fill=(0, 0, 0, 0),
        outline=(0, 0, 0, 30),
        width=shadow_margin
    )
    
    inner_shadow = inner_shadow.filter(ImageFilter.GaussianBlur(shadow_margin))
    
    # Create a subtle vignette effect
    vignette = create_radial_gradient(
        size, size, 
        (0, 0, 0, 0),        # Transparent center
        (0, 0, 0, 50)        # Dark edges with 20% opacity
    )
    
    # Apply inner shadow and vignette
    img.paste(inner_shadow, (0, 0), inner_shadow)
    img.paste(vignette, (0, 0), vignette)
    
    # Add subtle highlights along edges
    highlight = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    
    # Top-left highlight
    highlight_draw.line((0, 0, size//3, 0), fill=(255, 255, 255, 70), width=2)
    highlight_draw.line((0, 0, 0, size//3), fill=(255, 255, 255, 70), width=2)
    
    highlight = highlight.filter(ImageFilter.GaussianBlur(2))
    img.paste(highlight, (0, 0), highlight)
    
    # Apply rounded corners to the final image
    corner_radius = int(size * 0.2)  # Increased corner radius for modern look
    img = add_rounded_corners(img, corner_radius)
    
    # Add a drop shadow to the entire icon
    img = add_shadow(img, offset=(int(size*0.02), int(size*0.03)), blur_radius=20)
    
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
            try:
                resampling_filter = Image.LANCZOS
            except AttributeError:
                resampling_filter = Image.ANTIALIAS
        
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