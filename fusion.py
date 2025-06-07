import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import LidarSegPointCloud



nuscenes_path = "v1.0-mini"  # NuScenes dataset path
csv_path = "detections_real.csv"  # CSV with camera-detected objects
nuscenes = NuScenes(version='v1.0-mini', dataroot=nuscenes_path, verbose=True)


detections_df = pd.read_csv(csv_path)


# Function to process a sample
def compare_detections(sample_token):
    # Get the sample
    sample = nuscenes.get('sample', sample_token)

    # Process LiDAR data
    lidar_data_token = sample['data']['LIDAR_TOP']
    lidar_data = nuscenes.get('sample_data', lidar_data_token)

    # Load LiDAR points
    lidar_points = LidarPointCloud.from_file(nuscenes.dataroot + '/' + lidar_data['filename'])

    # Load LiDAR segmentation labels
    lidarseg_labels = nuscenes.get('lidarseg', lidar_data_token)
    lidarseg_filename = nuscenes.dataroot + '/' + lidarseg_labels['filename']
    labels = np.fromfile(lidarseg_filename, dtype=np.uint8)

    # Count unique segments (objects)
    unique_segments = np.unique(labels)
    num_objects_lidar = len(unique_segments)

    # Process Camera data and CSV
    num_objects_camera = 0
    for camera_channel in sample['data']:
        if 'CAM' in camera_channel:  # Loop over all cameras in the sample
            camera_data = nuscenes.get('sample_data', sample['data'][camera_channel])
            print(f"NuScenes Filename ({camera_channel}): {camera_data['filename']}")
            camera_filename = camera_data['filename']
            # Filter CSV for detections matching this filename
            matching_detections = detections_df[detections_df['Image Name'] == camera_filename]
            print(f"CSV Matches for {camera_channel}: {matching_detections.shape[0]}")
            num_objects_camera += matching_detections.shape[0]

    # Compare counts
    print(f"\nSample Token: {sample_token}")
    print(f"Number of objects detected by LiDAR: {num_objects_lidar}")
    print(f"Number of objects detected by camera: {num_objects_camera}")



# Example: Compare detections for the sample of choice
compare_detections(nuscenes.sample[250]['token'])