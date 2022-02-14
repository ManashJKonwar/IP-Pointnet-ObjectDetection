__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
from preprocessing.KITTI.generate_data import generate_data
from utility.utility_datatransformation import download_dataset
from training.KITTI.train_tf import train_frustum_pointnet_tf

if __name__ == '__main__':
    TRAIN_FRUSTUM_NET_KITTI = True
    TRAIN_FRUSTUM_NET_LYFT = False

    if TRAIN_FRUSTUM_NET_KITTI:
        '''
        Downloading KITTI Dataset
        Consists of following datasets:
            1. left color images
            2. velodyne point clouds
            3. camera calibration matrices
            4. training labels
        '''
        kitti_dataset_directory =  download_dataset(
                                        dataset_url_dict={
                                            'left color images':'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
                                            'velodyne point clouds':'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
                                            'camera calibration matrices':'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
                                            'training labels':'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
                                        },
                                        dataset_directory=os.getcwd(),
                                        extracted_folder_name=r'KITTI'
                                    )

        '''
        Data Preparation for KITTI Dataset
        During this process, Extraction of frustum point clouds along with ground truth labels 
        from the original KITTI data is performed which is based on both ground truth 2D bounding boxes 
        and boxes from a 2D object detector. We will do the extraction for the train dataset under 
        (datasets/KITTI/image_sets/train.txt) and validation sets under (datasets/KITTI/image_sets/val.txt)
        using ground truth 2D boxes, and also extract data from validation set with predicted 2D boxes 
        under (datasets/KITTI/rgb_detections/rgb_detection_val.txt).
        This is initially done for only 3 object types namely
        1. Car
        2. Pedestrain and 
        3. Cyclist
        However, the behaviour can be extended for remaining objects as well
        '''
        generate_data(
            dataset_directory=r'datasets\KITTI',
            generate_train_data=True,
            generate_val_data=True,
            generate_rgb_val_data=True,
            object_list=['Car', 'Pedestrian', 'Cyclist']
        )

        '''
        Training Frustum Pointnets
        '''
        train_frustum_pointnet_tf(
            gpu=0,
            model_name='frustum_pointnets_v1',
            log_dir=r'logs\KITTI',
            num_point=2048,
            max_epoch=201,
            batch_size=32,
            learning_rate=0.001,
            momentum=0.9,
            optimizer='adam',
            decay_step=200000,
            decay_rate=0.7,
            no_intensity=False
        )