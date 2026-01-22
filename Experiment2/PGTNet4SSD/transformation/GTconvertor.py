"""
@author: Keyvan Amiri Elyasi
"""

import os
import os.path as osp
import argparse
import yaml # type: ignore
import pickle
import random
import pm4py # type: ignore
from pm4py.objects.log.importer.xes import importer as xes_importer # type: ignore

from conversion_utils import read_user_inputs, case_full_func
from conversion_utils import train_test_split_func, eventlog_class_provider
from global_stat import get_global_stat_func
from core_function import graph_conversion_func
from clustering import apply_clustering


def main(directory, yml_file, overwrite):
    try:
        print("Starting main function...")
        print(f"Arguments: directory={directory}, yml_file={yml_file}, overwrite={overwrite}")
        trans_dir = os.getcwd()
        pgtnet_dir = os.path.dirname(trans_dir)
        root_directory = os.path.dirname(pgtnet_dir)

        # path to dataset folder
        datasets_directory = os.path.join(root_directory, "datasets")
        print(f"Datasets directory: {datasets_directory}")
        if not os.path.exists(datasets_directory):
            os.makedirs(datasets_directory)
        
        # read configuration file        
        yml_file_path = os.path.join(directory, yml_file)
        print(f"Checking if YAML file exists: {yml_file_path}")
        if not os.path.exists(yml_file_path):
            print("YAML file not found!")
            return
        user_inputs = read_user_inputs(yml_file_path)  
        print("YAML file parsed successfully.")
        
        # Check whether conversion is required or not.
        dataset_name = user_inputs.get('dataset_name')
        print(f"Dataset name: {dataset_name}")
        
        # Adjusted dataset path to include 'PGTNet4SSD'
        dataset_path = os.path.join(root_directory, 'PGTNet4SSD', 'raw_dataset', dataset_name)
        print(f"Dataset path: {dataset_path}")
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found at: {dataset_path}")
            return
        
        #dataset name: without .xes extension
        print("Check 1")
        dataset_name_no_ext = os.path.splitext(dataset_name)[0] 
        print("Check 2")
        graph_dataset_class_name = eventlog_class_provider(dataset_name_no_ext)
        print("Check 3")
        output_address_list = ['train.pickle', 'val.pickle', 'test.pickle']  
        graph_dataset_path =  os.path.join(datasets_directory, graph_dataset_class_name)
        graph_dataset_path_raw =  os.path.join(graph_dataset_path, "raw")
        graph_dataset_path_processed =  os.path.join(graph_dataset_path, "processed")        
        out_add0 = os.path.join(graph_dataset_path_raw, output_address_list[0])
        out_add1 = os.path.join(graph_dataset_path_raw, output_address_list[1])
        out_add2 = os.path.join(graph_dataset_path_raw, output_address_list[2])   
        print("Check 4")
        if (not overwrite and 
            os.path.exists(out_add0) and os.path.exists(out_add1) and os.path.exists(out_add2)):
            print(f"For event log: '{dataset_name_no_ext}' conversion is already done and overwrite is set to false.")
            print("Stopping the code.")
            return
        print("Check 5")
        dataset_path = os.path.join(root_directory, 'PGTNet4SSD\\raw_dataset', dataset_name)
        # Import the event log
        log = xes_importer.apply(dataset_path)   
        print("Check 6")
        event_log = pm4py.read_xes(dataset_path)   
        print("Check 7")
        # Assigning values from .yml file to global variables in main
        split_ratio = user_inputs.get('train_val_test_ratio')
        print("Check 8")
        event_attributes = user_inputs.get('event_attributes', [])
        event_num_att = user_inputs.get('event_num_att', [])
        case_attributes = user_inputs.get('case_attributes', [])
        case_num_att = user_inputs.get('case_num_att', [])
        target_normalization = user_inputs.get('target_normalization', True) 
        # add "case:" string to case attributes in .yml file
        case_attributes_full, case_num_ful = case_full_func(
            case_attributes,case_num_att) 
        # Split the event log into training, validation, and test sets
        print("Check 9")
        (train_df, val_df, test_df, train_val_df, train_log, val_log, test_log,
         train_val_log, max_time_norm, sorted_start_dates, sorted_end_dates
         ) = train_test_split_func(log, event_log, split_ratio)
        print("Check 10")
        # Get global information (only use training and validation sets)
        # one-hot encoding for categorical attributes
        (node_class_dict, max_case_df, max_active_cases, min_num_list,
         max_num_list, event_min_num_list, event_max_num_list, 
         attribute_encoder_list, case_encoder_list, node_dim, edge_dim, 
         avg_num_list, event_avg_num_list) = get_global_stat_func(
             train_val_log, train_val_df, case_num_ful, event_num_att,
             event_attributes, case_attributes_full, event_log, log)
        # Now the main part for converting prefixes into directed attributed graphs
        # a list to collect removed cases (any case with length less than 3)
        removed_cases = [] 
        idx = 0 # index for graphs
        # a list to collect all Pytorch geometric data objects.
        data_list = []
        print("Check 11")
        removed_cases, idx, data_list = graph_conversion_func(
            train_log, removed_cases, idx, data_list, case_attributes,
            case_encoder_list, case_num_att, min_num_list, max_num_list,
            event_attributes, event_num_att, target_normalization,
            max_time_norm, node_class_dict, edge_dim, max_case_df,
            sorted_start_dates, sorted_end_dates, max_active_cases,
            attribute_encoder_list, event_min_num_list, event_max_num_list,
            avg_num_list, event_avg_num_list)
        last_train_idx = idx
        print("Check 12")
        removed_cases, idx, data_list = graph_conversion_func(
            val_log, removed_cases, idx, data_list, case_attributes,
            case_encoder_list, case_num_att, min_num_list, max_num_list,
            event_attributes, event_num_att, target_normalization,
            max_time_norm, node_class_dict, edge_dim, max_case_df,
            sorted_start_dates, sorted_end_dates, max_active_cases,
            attribute_encoder_list, event_min_num_list, event_max_num_list,
            avg_num_list, event_avg_num_list)
        last_val_idx = idx
        print("Check 13")
        removed_cases, idx, data_list = graph_conversion_func(
            test_log, removed_cases, idx, data_list, case_attributes,
            case_encoder_list, case_num_att, min_num_list, max_num_list,
            event_attributes, event_num_att, target_normalization,
            max_time_norm, node_class_dict, edge_dim, max_case_df,
            sorted_start_dates, sorted_end_dates, max_active_cases,
            attribute_encoder_list, event_min_num_list, event_max_num_list,
            avg_num_list, event_avg_num_list)
        indices = list(range(len(data_list)))
        # data split based on the global graph list
        train_indices = indices[:last_train_idx]
        val_indices = indices[last_train_idx:last_val_idx]
        test_indices = indices[last_val_idx:] 
        data_train = [data_list[i] for i in train_indices]
        data_val = [data_list[i] for i in val_indices]
        data_test = [data_list[i] for i in test_indices]
        # shuffle the data in each split, to avoid order affect training process
        print("Check 14")
        random.shuffle(data_train)
        random.shuffle(data_val)
        random.shuffle(data_test)
        # Save the training, validation, and test datasets
        file_save_list = [data_train, data_val, data_test]      
        if not os.path.exists(graph_dataset_path):
            os.makedirs(graph_dataset_path)
        if not os.path.exists(graph_dataset_path_raw):
            os.makedirs(graph_dataset_path_raw)
        if not os.path.exists(graph_dataset_path_processed):
            os.makedirs(graph_dataset_path_processed)  
        for address_counter in range(len(output_address_list)):
            save_address = osp.join(graph_dataset_path_raw,
                                    output_address_list[address_counter])
            save_flie = open(save_address, "wb")
            pickle.dump(file_save_list[address_counter], save_flie)
            save_flie.close()
        # save important auxiliary information
        # apply clustering for cluster-based bucketing
        print("Check 15")
        cluster_counts = apply_clustering(graph_dataset_path_raw)
        edge_dim_path =  os.path.join(graph_dataset_path, "edge_dim.pkl")
        max_norm_path = os.path.join(graph_dataset_path, "max_time_norm.pkl")
        cluster_path = os.path.join(graph_dataset_path, "test_clusters.pkl")
        with open(edge_dim_path, 'wb') as file:
            pickle.dump(edge_dim, file)
        with open(max_norm_path, 'wb') as file:
            pickle.dump(max_time_norm, file)  
        with open(cluster_path, "wb") as f:
            pickle.dump(cluster_counts, f)
    # except FileNotFoundError:
    #     print("File not found. Please provide a valid file path.")
    except yaml.YAMLError as e:
        print("Error while parsing the .yml file.")
        print(e)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description='Converting event logs to graph datasets.')
    parser.add_argument('directory', type=str,
                        help="Directory of conversion configurations")
    parser.add_argument('yml_file', type=str, help='Name of the YAML file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Boolean indicating whether to overwrite')    
    args = parser.parse_args()
    print(args)
    main(args.directory, args.yml_file, args.overwrite)