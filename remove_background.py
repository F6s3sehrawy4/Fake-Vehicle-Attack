import os
from rembg import remove
import onnxruntime

# Define input and output directories of the vehicles wanted
input_dir = "Vehicles/Motorcycles"
output_dir = "Vehicles2/Motorcycles"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Ensure only image files are used
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_bg_removed.png")

        # Read the input image
        with open(input_path, "rb") as file:
            input_image = file.read()

        # Remove the background
        output_image = remove(input_image)

        # Save the output image
        with open(output_path, "wb") as file:
            file.write(output_image)

print("Background removal completed!")
