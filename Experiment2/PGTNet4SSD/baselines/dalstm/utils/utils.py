"""
@author: Keyvan Amiri Elyasi
"""
import os
import logging
import yaml
import sys
import random
import pickle
import numpy as np
import torch
import torch.optim as optim

##############################################################################
# Genral utility methods, and classes
##############################################################################
def get_logger(args, result_dir):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Train_Evaluation_Logger')
    logger.setLevel(logging.INFO)
    # Clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger_name = args.bucketing + '_report.log'
    logger_par_path = os.path.join(result_dir, logger_name)
    file_handler = logging.FileHandler(logger_par_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            sys.exit(f"Error reading YAML file: {e}")

def generate_seeds(num_seeds=5, base_seed=42, max_seed=10000):
    rng = random.Random(base_seed)  # deterministic generator
    return [rng.randint(0, max_seed) for _ in range(num_seeds)]  

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False        

# function to set the optimizer object
def set_optimizer (model, optimizer_type, base_lr, eps, weight_decay):
    if optimizer_type == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':   
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'Adam':   
        optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay) 
    # define scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)         
    return optimizer, scheduler

def delete_files(folder_path=None, substring=None, extension=None):
    files = os.listdir(folder_path)    
    for file in files:
        if (substring!= None) and (substring in file):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
        if (extension!= None) and (file.endswith(extension)):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            
def get_csv_paths(dataset_name, output_folder):
    full_dataset_name = dataset_name + '.csv'
    train_dataset_name = 'train_' + dataset_name + '.csv'
    val_dataset_name = 'val_' + dataset_name + '.csv'
    test_dataset_name = 'test_' + dataset_name + '.csv'
    full_dataset_path = os.path.join(output_folder, full_dataset_name)     
    train_dataset_path = os.path.join(output_folder, train_dataset_name)
    val_dataset_path = os.path.join(output_folder, val_dataset_name)
    test_dataset_path = os.path.join(output_folder, test_dataset_name) 
    return (full_dataset_path,train_dataset_path,val_dataset_path,test_dataset_path)

def get_tensor_paths(dataset_name, output_folder):
    X_train_path = os.path.join(
        output_folder, "DALSTM_X_train_"+dataset_name+".pt")
    X_val_path = os.path.join(
        output_folder, "DALSTM_X_val_"+dataset_name+".pt")
    X_test_path = os.path.join(
        output_folder, "DALSTM_X_test_"+dataset_name+".pt")
    y_train_path = os.path.join(
        output_folder, "DALSTM_y_train_"+dataset_name+".pt")
    y_val_path = os.path.join(
        output_folder, "DALSTM_y_val_"+dataset_name+".pt")
    y_test_path = os.path.join(
        output_folder, "DALSTM_y_test_"+dataset_name+".pt")
    return (X_train_path,X_val_path,X_test_path,y_train_path,y_val_path,y_test_path)

def get_length_paths(dataset_name, output_folder):
    train_length_path = os.path.join(
        output_folder, "DALSTM_train_length_list_"+dataset_name+".pkl") 
    val_length_path = os.path.join(
        output_folder, "DALSTM_val_length_list_"+dataset_name+".pkl")  
    test_length_path = os.path.join(
        output_folder, "DALSTM_test_length_list_"+dataset_name+".pkl") 
    return (train_length_path,val_length_path,test_length_path)

def get_id_paths(dataset_name, output_folder):
    train_id_path = os.path.join(
        output_folder, "DALSTM_train_case_list_"+dataset_name+".pkl") 
    val_id_path = os.path.join(
        output_folder, "DALSTM_val_case_list_"+dataset_name+".pkl") 
    test_id_path = os.path.join(
        output_folder, "DALSTM_test_case_list_"+dataset_name+".pkl") 
    return (train_id_path,val_id_path,test_id_path)

def get_cluster_paths(dataset_name, output_folder):
    train_cluster_path = os.path.join(
        output_folder, "DALSTM_train_cluster_list_"+dataset_name+".pkl") 
    val_cluster_path = os.path.join(
        output_folder, "DALSTM_val_cluster_list_"+dataset_name+".pkl") 
    test_cluster_path = os.path.join(
        output_folder, "DALSTM_test_cluster_list_"+dataset_name+".pkl") 
    return (train_cluster_path,val_cluster_path,test_cluster_path)

def get_shape_paths(dataset_name, output_folder):
    scaler_path = os.path.join(
        output_folder, "DALSTM_max_train_val_"+dataset_name+".pkl")
    input_size_path = os.path.join(
        output_folder, "DALSTM_input_size_"+dataset_name+".pkl")
    max_len_path = os.path.join(
        output_folder, "DALSTM_max_len_"+dataset_name+".pkl") 
    return (scaler_path, input_size_path, max_len_path)

def load_tensors(dataset_name, output_folder):
    (X_train_path,X_val_path,X_test_path,
     y_train_path,y_val_path,y_test_path) = get_tensor_paths(
         dataset_name, output_folder) 
    X_train = torch.load(X_train_path)
    X_val = torch.load(X_val_path)
    X_test = torch.load(X_test_path)
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_test = torch.load(y_test_path)
    return (X_train,X_val,X_test,y_train,y_val,y_test)

def load_length_lists(dataset_name, output_folder):
    (train_length_path,val_length_path,test_length_path) = get_length_paths(
        dataset_name, output_folder)
    with open(train_length_path, 'rb') as f:
        train_lengths =  pickle.load(f)
    with open(val_length_path, 'rb') as f:
        val_lengths =  pickle.load(f)
    with open(test_length_path, 'rb') as f:
        test_lengths =  pickle.load(f)
    return (train_lengths,val_lengths,test_lengths)

def load_cluster_lists(dataset_name, output_folder):
    (train_cluster_path,val_cluster_path,test_cluster_path) = get_cluster_paths(
        dataset_name, output_folder)
    with open(train_cluster_path, 'rb') as f:
        train_clusters =  pickle.load(f)
    with open(val_cluster_path, 'rb') as f:
        val_clusters =  pickle.load(f)
    with open(test_cluster_path, 'rb') as f:
        test_clusters =  pickle.load(f)    
    return (train_clusters,val_clusters,test_clusters)

def load_id_lists(dataset_name, output_folder):
    (train_id_path,val_id_path,test_id_path) = get_id_paths(
        dataset_name, output_folder)
    with open(train_id_path, 'rb') as f:
        train_ids =  pickle.load(f)
    with open(val_id_path, 'rb') as f:
        val_ids =  pickle.load(f)
    with open(test_id_path, 'rb') as f:
        test_ids =  pickle.load(f) 
    return (train_ids, val_ids, test_ids)

def get_shape_data(dataset_name, output_folder):
    (scaler_path, input_size_path, max_len_path) = get_shape_paths(
        dataset_name, output_folder)
    with open(scaler_path, 'rb') as f:
        max_train_val =  pickle.load(f)
    with open(input_size_path, 'rb') as f:
        input_size =  pickle.load(f)
    with open(max_len_path, 'rb') as f:
        max_len =  pickle.load(f) 
    return (max_train_val, input_size, max_len)

def load_case_mapping(dataset_name, output_folder):
    mapping_path = os.path.join(output_folder, dataset_name+"_mapping.pkl")
    with open(mapping_path, 'rb') as f:
        case_mapping =  pickle.load(f)
    return case_mapping