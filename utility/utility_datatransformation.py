__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import tqdm
import tensorflow as tf

def download_dataset(**kwargs):
    """
    This method downloads the dataset required for frustum nets from relevant url provided and 
    dumps this in the datasets folder
    Parameters: 
        dataset_url_dict (str): http links with relevant file types for downloading relevant datasets
    Returns: 
        None
    """
    dataset_url_dict = kwargs.get('dataset_url_dict')
    dataset_directory = kwargs.get('dataset_directory')
    extracted_folder_name = kwargs.get('extracted_folder_name')

    try:
        if dataset_url_dict is not None and len(dataset_url_dict.values())>0:
            # Check if dataset folder is present 
            if os.path.exists(os.path.join(dataset_directory, 'datasets', extracted_folder_name)):
                return os.path.join(dataset_directory, 'datasets', extracted_folder_name)
            
            data_directory_list=[]
            os.makedirs(os.path.join(dataset_directory, 'datasets', extracted_folder_name))

            # Download each dataset from links mentioned
            for link_key, link_val in tqdm.tqdm(dataset_url_dict.items(), desc='Downloading Datasets'):
                DATA_DIR = tf.keras.utils.get_file(
                    fname='_'.join(link_key.split(' '))+'.zip',
                    origin=link_val,
                    extract=False,
                    cache_dir=dataset_directory
                )
                DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), extracted_folder_name)
                data_directory_list.append(DATA_DIR)
            return data_directory_list
    except Exception:
        print('Caught Exception while downloading dataset', exc_info=True)