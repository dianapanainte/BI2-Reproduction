"""
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
#from pathlib import Path
#import random

from utils.bucket import create_partitions, get_partition_indices
from utils.bucket import get_ssd_dict, get_state
from utils.bucket import assert_bucketing, conduct_bucketing
from utils.model import DALSTMModel, train_model, test_model
from utils.utils import generate_seeds, set_random_seed
from utils.utils import load_tensors, load_length_lists, load_cluster_lists
from utils.utils import load_id_lists, get_shape_data, get_logger
from utils.utils import load_case_mapping, set_optimizer
  
def main():
    warnings.filterwarnings('ignore')
    # Parse arguments for training and inference
    parser = argparse.ArgumentParser(description='DALSTM Baseline')
    parser.add_argument('--dataset',
                        help='Raw dataset to predict remaining time for')    
    parser.add_argument('--bucketing', type=str, default='SSD_B',
                        choices=['N_B', 'L_B', 'C_B', 'SSD_B'],
                        help='Bucketing Strategy to be used.')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    args = parser.parse_args()
    # set device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'training and evaluation are done on: {device}')    
    # define important hyperparameters
    n_nuerons = 150
    n_layers = 2 # number of LSTM layers
    dropout = True # whether to apply dropout
    drop_prob = 0.2
    max_epochs = 200
    early_stop_patience = 20
    early_stop_min_delta = 0
    clip_grad_norm = None # value for norm clipping, None = no clipping
    clip_value = None # value for value clipping, None = no clipping  
    optimizer_type = 'NAdam'
    base_lr = 0.001 # base learning rate 
    eps = 1e-7 # epsilon parameter for Adam 
    weight_decay = 0.0  # weight decay for Adam 
    normalization = False # whether to normalized target attribute or not 
    # set important file names and paths
    dataset_name = args.dataset
    current_directory = os.getcwd()
    processed_data_path = os.path.join(current_directory, dataset_name)
    result_dir = os.path.join(current_directory, 'results', dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    # Add logger to keep track of important details
    logger = get_logger(args, result_dir)
    # load train-validation-test tensors
    (X_train,X_val,X_test,y_train,y_val,y_test) = load_tensors(
        dataset_name, processed_data_path)
    # load train-validation-test length lists
    (train_lengths,val_lengths,test_lengths) = load_length_lists(
        dataset_name, processed_data_path)
    # load train-validation-test cluster lists
    (train_clusters,val_clusters,test_clusters) = load_cluster_lists(
        dataset_name, processed_data_path)
    # load train-validation-test id lists
    (train_ids, val_ids, test_ids) = load_id_lists(
        dataset_name, processed_data_path)
    # load auxiliary information
    (max_train_val, input_size, max_len) = get_shape_data(
        dataset_name, processed_data_path)
    # create indices for multiple/single buckets
    train_lst, val_lst, test_lst = [], [], []
    if args.bucketing == 'N_B':
        logger.info('Results for No-bucketing.') 
        train_lst.append(list(range(len(train_ids))))
        val_lst.append(list(range(len(val_ids))))
        test_lst.append(list(range(len(test_ids))))
        assert_bucketing(train_ids, val_ids, test_ids,
                         train_lst, val_lst, test_lst)
    elif args.bucketing == 'L_B':
        all_lengths = train_lengths + val_lengths + test_lengths
        partitions = create_partitions(all_lengths)
        logger.info('Results for Length-bucketing.') 
        logger.info(f'Number of buckets: {len(partitions)}, Bucket length: {partitions}') 
        for partition in partitions:
            train_idx, val_idx, test_idx = get_partition_indices(
                partition, train_lengths, val_lengths, test_lengths)
            train_lst.append(train_idx)
            val_lst.append(val_idx)
            test_lst.append(test_idx)
        assert_bucketing(train_lengths, val_lengths, test_lengths,
                         train_lst, val_lst, test_lst)
    elif args.bucketing == 'C_B':
        unique_clusters = set(train_clusters + val_clusters + test_clusters)      
        clusters = [[item] for item in unique_clusters]
        logger.info('Results for Cluster-bucketing.') 
        logger.info(f'Number of buckets: {len(clusters)}, Cluster IDs: {clusters}') 
        for cluster in clusters:
            train_idx, val_idx, test_idx = get_partition_indices(
                cluster, train_clusters, val_clusters, test_clusters)
            train_lst.append(train_idx)
            val_lst.append(val_idx)
            test_lst.append(test_idx)
        assert_bucketing(train_clusters, val_clusters, test_clusters,
                         train_lst, val_lst, test_lst)
    else:
        logger.info('Results for SSD-bucketing.') 
        args.ssd_dir = os.path.join(current_directory, 'SSD_results')
        ssd_dict = get_ssd_dict(args)
        case_mapping = load_case_mapping(dataset_name, processed_data_path) 
        orig_train_ids = [case_mapping[int(i)] for i in train_ids]
        orig_val_ids = [case_mapping[int(i)] for i in val_ids]
        if args.dataset == 'Sepsis':
            orig_test_ids = [case_mapping.get(i, case_mapping[0]) for i in test_ids]
        else:
            orig_test_ids = [case_mapping[int(i)] for i in test_ids]
        train_states = get_state(ssd_dict, orig_train_ids)
        val_states = get_state(ssd_dict, orig_val_ids)
        test_states = get_state(ssd_dict, orig_test_ids)
        state_set = set(list(ssd_dict.keys()))
        logger.info(f'Existing states: {state_set}')
        logger.info('Number of prefixes in each state')
        for key in ssd_dict:
            logger.info(f'{key}, {len(ssd_dict[key])}')
        states = [[item] for item in state_set]       
        for state in states:
            train_idx, val_idx, test_idx = get_partition_indices(
                state, train_states, val_states, test_states)
            train_lst.append(train_idx)
            val_lst.append(val_idx)
            test_lst.append(test_idx)
        assert_bucketing(train_states, val_states, test_states,
                         train_lst, val_lst, test_lst)
        for i in range(len(train_lst)):
            logger.info(f'{len(train_lst[i])}, {len(val_lst[i])}, {len(test_lst[i])}')
    ##########################################################################
    # training and evaluation 
    ##########################################################################
    # generate random seeds    
    seeds = generate_seeds(args.num_seeds)  
    mae_lst, train_time_lst, test_time_lst = [], [], []
    for curr_seed in seeds:
        seed = int(curr_seed)
        set_random_seed(seed) 
        print('Pipline for:', dataset_name, 'seed:', seed)
        logger.info(f'Pipline for: {dataset_name}, Seed: {seed}') 
        num_bucket = len(train_lst)
        df_lst, inf_time_lst = [], []
        train_time = 0
        for bucket_idx in range(num_bucket):
            test_lengths_bucket = [test_lengths[i] for i in test_lst[bucket_idx]]
            # execute bucketing to get train, validation, test sets
            (train_dataset, val_dataset, test_dataset) = conduct_bucketing(
                X_train, X_val, X_test, 
                y_train, y_val, y_test,
                train_lst, val_lst, test_lst, 
                bucket_idx=bucket_idx)
            if len(test_lst[bucket_idx]) < 1:
                # no need for training if there is no test example in that bucket
                continue
            # define training, validation, test data loaders
            train_loader = DataLoader(train_dataset, batch_size=max_len, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=max_len, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=max_len, shuffle=False)
            # define loss function
            criterion = nn.L1Loss()
            # define the model
            model = DALSTMModel(input_size=input_size, hidden_size=n_nuerons,
                                        n_layers=n_layers, max_len=max_len,
                                        dropout=dropout, p_fix=drop_prob).to(device)
            # define optimizer
            optimizer, scheduler = set_optimizer(
                model, optimizer_type, base_lr, eps, weight_decay)
            # execute training:       
            bucket_time = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                device=device, num_epochs=max_epochs,
                early_patience=early_stop_patience, 
                min_delta=early_stop_min_delta,
                clip_grad_norm=clip_grad_norm, clip_value=clip_value, 
                result_dir=result_dir,
                bucket = args.bucketing, bucket_idx=bucket_idx,
                seed=seed, logger=logger)            
            # execute inference
            bucket_results, bucket_inf_time = test_model(
                model=model, test_loader=test_loader, 
                test_original_lengths=test_lengths_bucket,
                y_scaler=max_train_val, normalization=normalization,
                device=device, result_dir= result_dir,
                bucket=args.bucketing, bucket_idx=bucket_idx,
                seed=seed, logger=logger)
            train_time += bucket_time
            df_lst.append(bucket_results)
            inf_time_lst.append(bucket_inf_time)
        # aggregate results over all buckets
        result_df = pd.concat(df_lst, ignore_index=True)
        csv_filename = os.path.join(
            result_dir,
            '{}_seed_{}_inference_result.csv'.format(args.bucketing, seed))

        result_df.to_csv(csv_filename, index=False)
        avg_error = result_df["Absolute_error"].mean()
        mae_lst.append(avg_error)
        train_time_lst.append(train_time)
        avg_inf = sum(inf_time_lst) / len(inf_time_lst)
        test_time_lst.append(avg_inf)
    # aggregate results over all seeds
    logger.info(f'Mean Absolout Error for all seeds (days): {mae_lst}')
    average_mae = sum(mae_lst) / len(mae_lst)
    logger.info(f'Average Mean Absolout Error (days): {average_mae}')
    std_mae = np.std(mae_lst)
    logger.info(f'Standard deviation of Mean Absolout Error (days): {std_mae}')    
    avg_training_time = (sum(train_time_lst) / len(train_time_lst))/60
    average_inference_time = sum(test_time_lst) / len(test_time_lst)  
    logger.info(f'Training Time (minutes) : {avg_training_time}')         
    logger.info(f'Inference time for each instance (miliseconds): {average_inference_time}')


if __name__ == '__main__':
    main()