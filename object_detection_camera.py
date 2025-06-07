import os
import csv
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Use a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Folders containing the driving scenes captured by the cameras
#image_folders = ["v1.0-mini/samples", "v1.0-mini/sweeps"]
image_folders = ["fake_scenes/samples/CAM_BACK", "fake_scenes/samples/CAM_FRONT"]



# CSV file to store detections
output_csv = "detections.csv"


# Initialize the CSV file with headers
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Directory", "Class Name", "Confidence",
                     "Bounding Box (x1, y1, x2, y2)", "Sample Token"])

# Processing the images to extract detections
for folder in image_folders:
    for root, _, files in os.walk(folder):
        for image_name in files:
            if image_name.endswith((".jpg", ".jpeg", ".png")):  # Ensure it's an image file
                image_path = os.path.join(root, image_name)

                # Normalize the file path to use forward slashes for compatibility with filename in NuScenes API
                normalized_path = os.path.relpath(image_path, "v1.0-mini").replace("\\", "/")

                # Use the model to detect objects in the image
                result_predict = model.predict(source=image_path, imgsz=(640))


                # Check if result_predict is not empty
                if result_predict:
                    detections = []  # List to store detected objects

                    # Iterate through results
                    for result in result_predict:
                        for box in result.boxes:
                            class_id = int(box.cls)  # Class ID
                            class_name = result.names[class_id]  # Class name
                            confidence = box.conf  # Confidence score

                            # Bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            # Write detected object details to the CSV file
                            with open(output_csv, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(
                                    [normalized_path, os.path.basename(root), class_name,
                                     confidence.item(),
                                     f"({x1}, {y1}, {x2}, {y2})"])

                    # Print detected objects
                    print(f"Detected objects in {normalized_path}: {detections}")


                # Get the plot and convert to an image object
                plot = result.plot()

                # Display using matplotlib (useful if IPython's display isn't working)
                plt.imshow(plot)
                plt.axis('off')  # Hide axis
                plt.show()
            else:
                print(f"No predictions for {normalized_path}")
