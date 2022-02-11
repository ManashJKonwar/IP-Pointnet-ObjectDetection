__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
from utility.utility_datatransformation import download_dataset

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
        Train Data Preparation
        '''
        