"""
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import warnings
import torch

from utils.utils import read_config
from utils.preprocess import data_handling, dalstm_process

def main():
    warnings.filterwarnings('ignore')
    # Parse arguments for training and inference
    parser = argparse.ArgumentParser(description='DALSTM Baseline')
    parser.add_argument('--dataset',
                        help='Raw dataset to predict remaining time for')    
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--normalization', action='store_true', default=False, 
                        help='Whether to use normalization for targets')  
    args = parser.parse_args()
    # set device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'training and evaluation are done on: {device}')    
    # set important file names and paths
    dataset_name = args.dataset
    current_directory = os.getcwd()
    # get the directory that is two level higher than curret folder
    parent_directory = os.path.dirname(os.path.dirname(current_directory))
    raw_data_dir = os.path.join(parent_directory, 'raw_dataset')
    dataset_file = dataset_name+'.xes'
    path = os.path.join(raw_data_dir, dataset_file)
    processed_data_path = os.path.join(current_directory, dataset_name)
    preprocessing_cfg = read_config('preprocessing_config.yaml')  
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path) 
    # Conduct Pre-processing
    data_handling(xes=path, output_folder=processed_data_path, cfg=preprocessing_cfg)  
    dalstm_process(
        dataset_name=dataset_name, output_folder=processed_data_path,
        normalization=args.normalization) 
        
if __name__ == '__main__':
    main()