import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion

# Set matplotlib backend to ensure plots are displayed in PyCharm
matplotlib.use('TkAgg')

def project_lidar_to_image_from_directory(nusc, sample, image_dir, pointsensor_channel='LIDAR_TOP'):

    # Loop through each camera to project LiDAR points
    for cam_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:

        # Load camera data
        cam_token = sample['data'][cam_channel]  # Token of the camera
        cam_data = nusc.get('sample_data', cam_token)  # Camera details
        cam_cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])  # Calibration details token

        # Camera intrinsic matrix
        intrinsic = np.array(cam_cs_record['camera_intrinsic'])  # Intrinsic: how the camera sees the world

        # Get the path for the image and ensure it loads correctly
        img_path = image_dir + '/' + cam_data['filename']
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error loading image for {cam_channel}: {e}")
            continue  # Skip to the next camera if there's an issue with loading the image

        # Load LiDAR data
        lidar_token = sample['data'][pointsensor_channel]
        print("LiDAR token:", lidar_token)
        lidar_data = nusc.get('sample_data', lidar_token)
        print("LiDAR data:", lidar_data)
        lidar_cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        print("LiDAR calibration record:", lidar_cs_record)
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
        print("LiDAR point cloud shape:", pc.points.shape)

        # Extrinsic transformation: transform LiDAR points to the camera coordinates
        lidar_rotation = Quaternion(lidar_cs_record['rotation'])  # LiDAR rotation quaternion
        cam_rotation = Quaternion(cam_cs_record['rotation'])  # Camera rotation quaternion

        # Calculate transformation from LiDAR to camera coordinates
        lidar_to_cam_transformation = transform_matrix(
            cam_cs_record['translation'], cam_rotation, inverse=True  # Transforms from ego coord to camera coord
        ).dot(transform_matrix(  # Combines those two matrices to transform from lidar to ego then to camera
            lidar_cs_record['translation'], lidar_rotation  # Transforms from lidar to ego
        ))

        # Apply transformation to LiDAR points
        pc_points_cam = lidar_to_cam_transformation.dot(np.vstack((pc.points[:3, :], np.ones(pc.points.shape[1]))))  # Each column in this matrix represents a point in homogeneous coordinates (x, y, z, 1)

        # Project the transformed points onto the image plane
        points_in_img = view_points(pc_points_cam[:3, :], intrinsic, normalize=True)

        # Filter points that are in front of the camera and within image bounds
        mask = np.logical_and(points_in_img[2, :] > 0,
                              np.logical_and(0 <= points_in_img[0, :], points_in_img[0, :] < cam_data['width']),
                              np.logical_and(0 <= points_in_img[1, :], points_in_img[1, :] < cam_data['height']))

        # Display the image with the LiDAR points overlaid
        plt.figure(figsize=(9, 5))
        plt.imshow(img)
        if mask.any():
            plt.scatter(points_in_img[0, mask], points_in_img[1, mask], c=pc_points_cam[2, mask], s=1, cmap='viridis')
        else:
            print(f"No LiDAR points within image bounds for {cam_channel}")

        plt.title(f"LiDAR Points Projected on {cam_channel}")
        plt.axis('off')
        plt.show()

# Example call to the function
def main():
    # Initialize NuScenes object
    nusc = NuScenes(version='v1.0-mini', dataroot='v1.0-mini')  # Replace with your dataset path

    # Select a sample
    sample = nusc.sample[15]

    # Define the directory where the scenes are stored
    image_dir = 'dataset/test/fake'  # Replace with your image directory

    # Call the function to project LiDAR points onto the images
    project_lidar_to_image_from_directory(nusc, sample, image_dir)

if __name__ == "__main__":
    main()
