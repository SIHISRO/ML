
from PIL import Image, ImageEnhance
import random
import os


def process_image_randomly(input_path, output_path, min_scale=0.8, max_scale=1.2, max_angle=45):
    """
    Randomly rotates and scales an image, optionally enhances lighting, and saves it.
    
    :param input_path: Path to the input image.
    :param output_path: Path where the processed image will be saved.
    :param min_scale: Minimum scaling factor.
    :param max_scale: Maximum scaling factor.
    :param max_angle: Maximum random rotation angle in degrees.
    """
    # 1. Open the image
    with Image.open(input_path) as img:
        # 2. Apply Random Scaling
        scale_factor = random.uniform(min_scale, max_scale)
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        # Resize using LANCZOS for high-quality downscaling/upscaling
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 3. Apply Random Rotation
        angle = random.uniform(-max_angle, max_angle)
        # expand=True ensures the entire image fits in the new canvas without cropping
        img_rotated = img_resized.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        # 4. Optional: Adjust Lighting (Brightness)
        # Enhance brightness by a random factor between 0.8 and 1.2
        enhancer = ImageEnhance.Brightness(img_rotated)
        brightness_factor = random.uniform(0.8, 1.2)
        img_final = enhancer.enhance(brightness_factor)
        
        # 5. Save the image
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_path):
            os.makedirs(output_dir)
            
        img_final.save(output_path)
        return output_path

# Example Usage:
# input_img = "path/to/your/image.jpg"
# output_img = "path/to/output/rotated_scaled.jpg"
# process_image_randomly(input_img, output_img)   



process_image_randomly(input_path="car.jpg",output_path="car2.jpg")