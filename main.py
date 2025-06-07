import pandas as pd
from nuscenes.nuscenes import NuScenes

# Load nuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='v1.0-mini')

# Load camera detections from CSV
camera_detections = pd.read_csv('detections.csv')

# Retrieve a sample and its LiDAR data
sample = nusc.sample[0]  # Example: first sample
lidar_token = sample['data']['LIDAR_TOP']

# Filter the camera detections to only those related to the current sample
# Assuming 'sample_token' exists in your CSV file and corresponds to nuScenes sample_token
camera_detections_for_sample = camera_detections[camera_detections['Sample Token'] == sample['token']]

# Get the number of detected objects in camera detections for the current sample
num_camera_objects = len(camera_detections_for_sample)

# Retrieve annotations for the LiDAR
lidar_data = nusc.get('sample_data', lidar_token)
lidar_annotations = [ann for ann in nusc.sample_annotation if ann['sample_token'] == lidar_data['sample_token']]

# Get the number of detected objects in LiDAR for the current sample
num_lidar_objects = len(lidar_annotations)

# Compare the number of detected objects
if num_camera_objects != num_lidar_objects:
    print(f"Mismatch in detected objects: ")
    print(f" - Camera detected {num_camera_objects} objects")
    print(f" - LiDAR detected {num_lidar_objects} objects")
else:
    print(f"Number of detected objects is the same: {num_camera_objects}")
