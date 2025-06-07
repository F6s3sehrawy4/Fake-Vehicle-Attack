from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix

# Load the dataset (adjust path to where you've stored the dataset)
nusc = NuScenes(version='v1.0-mini', dataroot='dataSets/v1.0-mini')
# nuscenes-lidarseg
nusc.list_lidarseg_categories(sort_by='count')
# Panoptic nuScenes
#nusc.list_lidarseg_categories(sort_by='count', gt_from='panoptic')
#instance statistics of panoptic dataset
#nusc.list_panoptic_instances(sort_by='count')
#sample chosen
my_sample = nusc.sample[87]
#Now let's take a look at what classes are present in the pointcloud of this particular sample
# nuscenes-lidarseg
#By doing sort_by='count', the classes and their respective frequency counts are printed in ascending order
#you can also do sort_by='name' and sort_by='index'
nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count')
# Panoptic nuScenes
#nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count', gt_from='panoptic')
#rendering in BEV lidarseg
sample_data_token = my_sample['data']['LIDAR_TOP']
nusc.render_sample_data(sample_data_token,
                        with_anns=False,
                        show_lidarseg=True,
                        show_lidarseg_legend=True)
#rendering in BEV with filtering
nusc.render_sample_data(sample_data_token,
                        with_anns=False,
                        show_lidarseg=True,
                        filter_lidarseg_labels=[22, 23])
#render on image
# nuscenes-lidarseg
nusc.render_pointcloud_in_image(my_sample['token'],
                                pointsensor_channel='LIDAR_TOP',
                                camera_channel='CAM_BACK',
                                render_intensity=False,
                                show_lidarseg=True,
                                show_lidarseg_legend=True)













