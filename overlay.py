import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil

# Paths to NuScenes dataset and Adversarial vehicles
scene_dataset_path = "dataset/train/real"
vehicle_dataset_path = "Vehicles2"
output_path = "dataset/test/fake"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)


# Function to add a vehicle to a scene image with realistic adjustments
def overlay_vehicle(scene_image, vehicle_dir_path):
    # Randomly choose a class folder
    class_folder = random.choice([
        os.path.join(vehicle_dir_path, class_name)
        for class_name in os.listdir(vehicle_dir_path)
        if os.path.isdir(os.path.join(vehicle_dir_path, class_name))
    ])
    class_name = os.path.basename(class_folder)

    # Randomly choose a vehicle image from the selected class folder
    vehicle_image = random.choice([
        os.path.join(class_folder, file)
        for file in os.listdir(class_folder)
        if file.endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Load the images
    scene = Image.open(scene_image)
    vehicle = Image.open(vehicle_image).convert("RGBA")

    # Resize the vehicle if it's too large to fit in the scene
    vehicle = resize_vehicle_to_scene(vehicle, scene.width, scene.height)

    # Resize vehicle based on a scale factor relative to the scene
    scale_factor = random.uniform(1, 1.5)  # Adjust range for realistic scaling
    vehicle_width, vehicle_height = vehicle.size
    new_vehicle_size = (int(vehicle_width * scale_factor), int(vehicle_height * scale_factor))
    vehicle = vehicle.resize(new_vehicle_size, Image.LANCZOS)

    # Position the vehicle at a random location near the bottom half of the scene
    max_x = max(scene.width - vehicle.width, 0)
    min_y = int(scene.height * 0.5)
    max_y = max(scene.height - vehicle.height, min_y + 1)
    if max_x == 0 or max_y <= min_y:
        print(f"Skipping overlay: Invalid position range for vehicle placement.")
        return scene

    position = (random.randint(0, max_x), random.randint(min_y, max_y))

    # Adjust brightness and contrast to match the scene
    enhancer = ImageEnhance.Brightness(vehicle)
    vehicle = enhancer.enhance(random.uniform(0.8, 1.2))  # Adjust brightness
    enhancer = ImageEnhance.Contrast(vehicle)
    vehicle = enhancer.enhance(random.uniform(0.8, 1.2))  # Adjust contrast

    # Apply blur if the vehicle is distant to enforce realism
    if position[1] > scene.height * 0.75:  # More blur if farther in the background
        vehicle = vehicle.filter(ImageFilter.GaussianBlur(2))

    # Add a shadow for depth
    shadow = vehicle.copy().convert("RGBA")
    shadow = shadow.point(lambda p: p * 0)  # Make the shadow black
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))  # Blur shadow
    shadow_position = (position[0] + 10, position[1] + 10)  # Offset shadow

    # Overlay the shadow and vehicle on the scene
    scene.paste(shadow, shadow_position, shadow)
    scene.paste(vehicle, position, vehicle)

    return scene


# Validate and resize the vehicle if it exceeds scene dimensions
def resize_vehicle_to_scene(vehicle, scene_width, scene_height):
    vehicle_width, vehicle_height = vehicle.size

    # If the vehicle is larger than the scene, resize it to fit
    if vehicle_width > scene_width or vehicle_height > scene_height:
        # Calculate the scaling factor to fit the vehicle into the scene
        width_scale = scene_width / vehicle_width
        height_scale = scene_height / vehicle_height
        scale_factor = min(width_scale, height_scale)  # Choose the smaller scale to maintain aspect ratio

        # Resize vehicle to fit the scene
        new_vehicle_size = (int(vehicle_width * scale_factor), int(vehicle_height * scale_factor))
        vehicle = vehicle.resize(new_vehicle_size, Image.LANCZOS)

    return vehicle


# Iterate over scene images
scene_images = os.listdir(scene_dataset_path)

for i, scene_filename in enumerate(scene_images):
    scene_image_path = os.path.join(scene_dataset_path, scene_filename)

    # Save the original image to the dataset/test/real directory
    original_image_path = os.path.join('dataset/test/real', scene_filename)
    shutil.copy(scene_image_path, original_image_path)  # Copy the original image

    # Overlay vehicle on the scene
    result_image = overlay_vehicle(scene_image_path, vehicle_dataset_path)

    # Save the resulting image
    result_filename = scene_filename
    result_image.save(os.path.join(output_path, result_filename))

print("Vehicles have been overlaid and saved to the output folder.")
