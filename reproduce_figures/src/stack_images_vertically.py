from PIL import Image
import os
import numpy as np

def crop_transparency(img):
    """Remove transparent background from an RGBA image and replace with white."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')  # Ensure image has an alpha channel
    
    img_array = np.array(img)  # Convert to NumPy array
    alpha_channel = img_array[:, :, 3]  # Extract alpha channel
    
    # Find bounding box of non-transparent pixels
    non_transparent = np.where(alpha_channel > 0)  
    if non_transparent[0].size == 0:  
        return img  # Return original if fully transparent
    
    y_min, y_max = np.min(non_transparent[0]), np.max(non_transparent[0])
    x_min, x_max = np.min(non_transparent[1]), np.max(non_transparent[1])
    
    # Crop image
    cropped_img = img.crop((x_min, y_min, x_max, y_max))

    # Convert transparent pixels to white
    white_bg = Image.new("RGBA", cropped_img.size, (255, 255, 255, 255))
    white_bg.paste(cropped_img, (0, 0), cropped_img)

    return white_bg.convert("RGB")  # Convert to RGB to remove alpha channel

def stack_images_vertically(image_paths, output_path):
    """
    Stacks multiple images vertically, scaling the second image to match the width of the first.
    
    Parameters:
    - image_paths (list): List of image file paths.
    - output_path (str): Path to save the combined image.
    """
    images = [crop_transparency(Image.open(img)) for img in image_paths]

    # Use the first image as the reference size
    base_width = images[0].width

    # Resize other images to match the width while maintaining aspect ratio
    resized_images = [images[0]]
    for img in images[1:]:
        aspect_ratio = img.height / img.width
        new_height = int(base_width * aspect_ratio)
        resized_images.append(img.resize((base_width, new_height), Image.LANCZOS))

    # Determine total height
    new_height = sum(img.height for img in resized_images)

    # Create a blank image with a white background
    combined_img = Image.new("RGB", (base_width, new_height), "white")

    # Paste images
    y_offset = 0
    for img in resized_images:
        combined_img.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the combined image
    combined_img.save(output_path)
    return combined_img
