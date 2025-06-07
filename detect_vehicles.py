import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Folders containing the driving scenes captured by the cameras
image_folders = [
    "fake_scenes/single"
]
output_dir = "processed_scenes_fake"

# Process images in the specified folders
for folder in image_folders:
    # Extract folder name to create a corresponding output subdirectory
    folder_name = os.path.basename(folder)
    folder_output_dir = os.path.join(output_dir, folder_name)

    # Create the output subdirectory if it doesn't exist
    os.makedirs(folder_output_dir, exist_ok=True)

    for root, _, files in os.walk(folder):
        for image_name in files:
            if image_name.endswith((".jpg", ".jpeg", ".png")):  # Ensure it's an image file
                image_path = os.path.join(root, image_name)

                # Use the model to detect objects in the image
                results = model.predict(source=image_path, imgsz=640)

                # Save the results for each image
                for idx, result in enumerate(results):
                    # Get the processed image
                    plot = result.plot()

                    # Construct output path
                    output_path = os.path.join(folder_output_dir, f"{os.path.splitext(image_name)[0]}_processed_{idx}.png")

                    # Save the processed image
                    plt.imsave(output_path, plot)

print("Processing completed. Processed images are saved in", output_dir)
