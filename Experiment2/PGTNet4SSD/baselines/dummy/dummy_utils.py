# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:58:12 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

def split_log(log, train_ratio=0.8, 
              case_col ='case:concept:name', time_col='time:timestamp'):
    case_start_times = log.groupby(case_col)[time_col].min()
    sorted_case_ids = case_start_times.sort_values().index.tolist()
    split_index = int(len(sorted_case_ids) * train_ratio)
    train_case_ids = sorted_case_ids[:split_index]
    test_case_ids = sorted_case_ids[split_index:]
    train_df = log[log[case_col].isin(train_case_ids)].copy()
    test_df = log[log[case_col].isin(test_case_ids)].copy()
    return (train_case_ids, test_case_ids, train_df, test_df)

def add_rem_time(df_inp,
                 case_col ='case:concept:name', time_col='time:timestamp'):
    df = df_inp.copy()
    df['rem_time'] = df.groupby(case_col)[time_col].transform(
        lambda x: (x.max() - x).dt.total_seconds()
        )
    return df

def add_prefix_length(df_inp,
                      case_col ='case:concept:name', time_col='time:timestamp'):
    df = df_inp.copy()
    df = df.sort_values([case_col, time_col])
    df['prefix_length'] = df.groupby(case_col).cumcount() + 1
    return df

def create_length_partitions(df_inp, threshold_ratio=0.1,
                             case_col ='case:concept:name'):
    df1 = df_inp[df_inp.groupby(case_col)['prefix_length'].transform('max') 
                      != df_inp['prefix_length']].copy()
    df = df1[df1['prefix_length'] != 1].copy()  
    length_counts = df['prefix_length'].value_counts().sort_index()
    min_count = len(df) * threshold_ratio
    partitions = []
    current_range = []
    current_total = 0    
    for length, count in length_counts.items():
        if count >= min_count:
            if current_range:
                partitions.append(current_range)
                current_range = []
            partitions.append([length])
        else:
            current_range.append(length)
            current_total += count
            if current_total >= min_count:
                partitions.append(current_range)
                current_range = []
                current_total = 0    
    # Add all remaining lengths as one final partition
    if current_range:
        partitions.append(current_range)    
    return partitions

def find_closest_partition_index(length, partitions):
    return min(range(len(partitions)), key=lambda i: min(abs(length - x) for x in partitions[i]))

def no_bucket_dummy(train_df, test_df, case_col ='case:concept:name'):
    train1 = train_df[train_df.groupby(case_col)['prefix_length'].transform('max') 
                      != train_df['prefix_length']].copy()
    sel_train = train1[train1['prefix_length'] != 1].copy()    
    mean_rem = sel_train['rem_time'].mean()
    test1 = test_df[test_df.groupby(case_col)['prefix_length'].transform('max') 
                      != test_df['prefix_length']].copy()
    sel_test = test1[test1['prefix_length'] != 1].copy()    
    mae_test = (sel_test['rem_time'] - mean_rem).abs().mean()/3600/24
    return mae_test

def length_bucket_dummy(train_df, test_df, partitions,
                        case_col ='case:concept:name'):
    train1 = train_df[train_df.groupby(case_col)['prefix_length'].transform('max') 
                      != train_df['prefix_length']].copy()
    sel_train = train1[train1['prefix_length'] != 1].copy()  
    mean_lst = []
    for partition in partitions:
        df = sel_train[sel_train['prefix_length'].isin(partition)]
        mean_rem = df['rem_time'].mean()
        mean_lst.append(mean_rem)
    test1 = test_df[test_df.groupby(case_col)['prefix_length'].transform('max') 
                          != test_df['prefix_length']].copy()
    sel_test = test1[test1['prefix_length'] != 1].copy() 
    sel_test['prediction'] = sel_test['prefix_length'].apply(
    lambda x: mean_lst[find_closest_partition_index(x, partitions)])
    mae_test = (sel_test['prediction'] - sel_test['rem_time']).abs().mean() / (24 * 3600)    
    return mae_test   

def get_ssd_dict(args):
    ssd_dir = os.path.join(args.current_dir, 'SSD_results')
    dataset_name = os.path.splitext(args.dataset)[0]
    if dataset_name == 'BPIC12':
        ssd_name = dataset_name + '_D_SS.pkl'
    else:
        ssd_name = dataset_name + '_7D_SS.pkl'
    with open(os.path.join(ssd_dir, ssd_name), 'rb') as file:
        ssd_dict = pickle.load(file)
    if dataset_name == 'HelpDesk': 
        merged_list = []
        for k in ssd_dict:
            if k != 0:
                merged_list.extend(ssd_dict[k])
        out_dict = {0: ssd_dict[0], 1: merged_list}
    else:
        out_dict = ssd_dict
    return out_dict

def ssd_bucket_dummy(train_df, test_df, ssd_dict,
                        case_col ='case:concept:name'):
    # remove first and last prefixes
    train1 = train_df[train_df.groupby(case_col)['prefix_length'].transform('max') 
                      != train_df['prefix_length']].copy()
    sel_train = train1[train1['prefix_length'] != 1].copy() 
    sel_train[case_col] = sel_train[case_col].astype(str)
    test1 = test_df[test_df.groupby(case_col)['prefix_length'].transform('max') 
                          != test_df['prefix_length']].copy()
    sel_test = test1[test1['prefix_length'] != 1].copy() 
    sel_test[case_col] = sel_test[case_col].astype(str)
    mae_lst = []
    for key in ssd_dict:
        state_cases = ssd_dict[key]
        df_train = sel_train[sel_train[case_col].isin(state_cases)]
        df_test = sel_test[sel_test[case_col].isin(state_cases)]
        #print(key, len(df_train), len(df_test))
        if len(df_test) > 0:
            mean_rem = df_train['rem_time'].mean()
            mae_key = (mean_rem - df_test['rem_time']).abs().mean() / (24 * 3600)
            mae_lst.append((mae_key, len(df_test)))  
    total_frequency = sum(freq for _, freq in mae_lst)
    mae_test = sum(error * freq for error, freq in mae_lst) / total_frequency
    return mae_test 

def decision_tree_clustering(df, train_case_ids, test_case_ids, selected_cols,
                             case_col ='case:concept:name', target_col='rem_time',
                             threshold=0.1, handle_unknown='default'):
    categorical_cols = []
    for col in selected_cols:
        if col in {'case:AMOUNT_REQ', 'case:SUMleges', 'prefix_length',
                   'elapsed_time', 'processing_time'}:
            continue
        else:
            categorical_cols.append(col)
    df_working = df.copy()
    train_mask = df_working[case_col].isin(train_case_ids)
    test_mask = df_working[case_col].isin(test_case_ids)   
    train_df = df_working[train_mask].copy()
    test_df = df_working[test_mask].copy()
    X_train = train_df[selected_cols].copy()
    X_test = test_df[selected_cols].copy()   
    X_all = df_working[selected_cols].copy()    
    min_samples = int(threshold * len(X_test))   
    # Encode categorical variables
    label_encoders = {}
    for col in selected_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            test_values = X_test[col].astype(str)
            train_classes = set(le.classes_)
            mask_unknown = ~test_values.isin(train_classes)
            if mask_unknown.any():
                if handle_unknown == 'default':
                    # Map unknown values to -1
                    X_test[col] = test_values.map(lambda x: le.transform([x])[0] if x in train_classes else -1)
                else:
                    # Alternative: use most frequent value
                    most_frequent = le.classes_[0]
                    X_test[col] = test_values.map(lambda x: le.transform([x])[0] if x in train_classes else le.transform([most_frequent])[0])
            else:
                X_test[col] = le.transform(test_values)
            # Handle all data for clustering
            all_values = X_all[col].astype(str)
            mask_unknown_all = ~all_values.isin(train_classes)
            if mask_unknown_all.any():
                X_all[col] = all_values.map(lambda x: le.transform([x])[0] if x in train_classes else -1)
            else:
                X_all[col] = le.transform(all_values)
            label_encoders[col] = le  
    # Prepare target variable
    y_train = train_df[target_col]
    # Train decision tree regressor
    dt_model = DecisionTreeRegressor(min_samples_leaf=min_samples, random_state=42)    
    dt_model.fit(X_train, y_train)    
    # Use decision tree to assign clusters (leaf indices)
    leaf_indices = dt_model.apply(X_all)
    df_working['cluster'] = leaf_indices
    unique_leaves = df_working['cluster'].unique()
    leaf_to_cluster = {leaf: i for i, leaf in enumerate(sorted(unique_leaves))}
    df_working['cluster'] = df_working['cluster'].map(leaf_to_cluster)
    
    # Print cluster information
    cluster_counts = df_working['cluster'].value_counts().sort_index()
    print(f"\nNumber of clusters: {len(cluster_counts)}")
    print("Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples")    
    return df_working, dt_model

def clustering_columns(args):
    dataset_name = os.path.splitext(args.dataset)[0]
    if dataset_name in {'BPIC15_1', 'BPIC15_2', 'BPIC15_3', 'BPIC15_4', 'BPIC15_5'}:
        case_attributes = [
            'case:caseStatus', 'case:last_phase', 'case:Responsible_actor', 
            'case:parts', 'case:termName', 'case:requestComplete', 
            'case:caseProcedure', 'case:Includes_subCases', 'case:SUMleges'] 
    elif dataset_name == 'HelpDesk':
        case_attributes = ['case:responsible_section', 'case:support_section',
                           'case:product']
    elif dataset_name == 'BPIC12':
        case_attributes = ['case:AMOUNT_REQ']
    else:
        case_attributes = []
    additional_cols = ['prefix_length', 'elapsed_time', 'processing_time', 'concept:name']
    case_attributes.extend(additional_cols)    
    return case_attributes

def get_clusters(args, df_inp, train_case_ids, test_case_ids,
                 case_col='case:concept:name', time_col='time:timestamp'):
    df = df_inp.copy()
    df = df.sort_values([case_col, time_col])
    df['elapsed_time'] = df.groupby(case_col)[time_col].transform(
        lambda x: (x - x.min()).dt.total_seconds())   
    df['processing_time'] = df.groupby(case_col)[time_col].transform(
        lambda x: (x - x.shift(1)).dt.total_seconds())
    df['processing_time'] = df['processing_time'].fillna(0)
    selected_cols = clustering_columns(args)
    df_new, _ = decision_tree_clustering(
        df, train_case_ids, test_case_ids, selected_cols)
    clusters = list(set(list(df_new['cluster'].values)))
    return df_new, clusters

def clustering_bucket_dummy(df_inp, train_case_ids, test_case_ids, clusters,
                            case_col ='case:concept:name'):
    # remove first and last prefixes
    df1 = df_inp[df_inp.groupby(case_col)['prefix_length'].transform('max') 
                      != df_inp['prefix_length']].copy()
    df = df1[df1['prefix_length'] != 1].copy() 
    df[case_col] = df[case_col].astype(str)
    train_mask = df[case_col].isin(train_case_ids)
    test_mask = df[case_col].isin(test_case_ids)   
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    mae_lst = []
    for cluster in clusters:
        df_train = train_df[train_df['cluster'] == cluster]
        df_test = test_df[test_df['cluster'] == cluster]
        if len(df_test) > 0:       
            mean_rem = df_train['rem_time'].mean()
            mae_key = (mean_rem - df_test['rem_time']).abs().mean() / (24 * 3600)
            mae_lst.append((mae_key, len(df_test)))  
    total_frequency = sum(freq for _, freq in mae_lst)
    mae_test = sum(error * freq for error, freq in mae_lst) / total_frequency
    return mae_test 