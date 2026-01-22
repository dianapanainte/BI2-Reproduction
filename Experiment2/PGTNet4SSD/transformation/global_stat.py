# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:56:07 2025
@author: Keyvan Amiri Elyasi
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from conversion_utils import ActiveCase

# Get other relevant global information from training data + required one hot encoders
def get_global_stat_func(training_validation_log, training_validation_df,
                         case_num_ful, event_num_att, event_attributes,
                         case_attributes_full, event_log, log):
    """
    node_class_dict (activity classes), keys: tuple of activity/lifecycle, value: integer representation
    max_case_df: maximum number of same df relationship in one case
    max_active_cases: maximum number of concurrent active cases in training-validation sets
    """
    max_case_df = 0  
    node_class_dict = {}
    node_class_rep = 0
    start_dates, end_dates = [], []
    # Get node_class_dict, max_case_df
    for case_counter in range (len(log)):
        df_dict = {} # Dict: all activity-class df relationships and their frequencies
        current_case = log[case_counter]
        case_length = len(current_case)
        if case_length > 1:
            # iterate over all events of the case, collect df information
            for event_counter in range(case_length-1): 
                source_class = (current_case[event_counter].get('concept:name'), 
                                current_case[event_counter].get('lifecycle:transition'))
                target_class = (current_case[event_counter+1].get('concept:name'), 
                                current_case[event_counter+1].get('lifecycle:transition'))
                df_class = (source_class, target_class)
                if df_class in df_dict:
                    df_dict[df_class] += 1
                else:
                    df_dict[df_class] = 1
                if not (source_class in node_class_dict):
                    node_class_dict[source_class] = node_class_rep
                    node_class_rep += 1                                 
            if max((df_dict).values()) > max_case_df:
                max_case_df = max((df_dict).values())    
    # Iterate over train-val data, get list of start and end dates, use them to get max_active_cases
    for case_counter in range (len(training_validation_log)):
        current_case = log[case_counter]
        case_length = len(current_case)
        start_dates.append(current_case[0].get('time:timestamp'))
        end_dates.append(current_case[case_length-1].get('time:timestamp'))          
    sorted_start_dates = sorted(start_dates)
    sorted_end_dates = sorted(end_dates)
    max_active_cases = 0
    unique_timestamps = list(training_validation_df['time:timestamp'].unique())
    for any_time in unique_timestamps:
        cases_in_system = ActiveCase(sorted_start_dates, sorted_end_dates, any_time)
        if cases_in_system > max_active_cases:
            max_active_cases = cases_in_system    
    # obtain number of numerical case attributes + 3 lists for min/max/avg values for each attribute
    min_num_list, max_num_list, avg_num_list = [], [], []
    case_num_card = len(case_num_ful)
    for num_att in case_num_ful:
        unique_values = training_validation_df[num_att].dropna().tolist()
        unique_values_float = [float(val) for val in unique_values]
        min_num_list.append(float(min(unique_values_float)))
        max_num_list.append(float(max(unique_values_float)))
        avg_num_list.append(float(sum(unique_values_float)/len(unique_values_float)))
    # obtain number of numerical event attributes + 3 lists for min/max/avg values for each attribute
    event_min_num_list, event_max_num_list, event_avg_num_list = [], [], []
    event_num_card = len(event_num_att)
    for num_att in event_num_att:
        unique_values = training_validation_df[num_att].dropna().tolist()
        unique_values_float = [float(val) for val in unique_values]
        event_min_num_list.append(float(min(unique_values_float)))
        event_max_num_list.append(float(max(unique_values_float)))
        event_avg_num_list.append(float(sum(unique_values_float)/len(unique_values_float))) 
    # List of one-hot encoders: categorical event attributes of intrest
    attribute_encoder_list = []
    attribute_cardinality = 0
    for event_attribute in event_attributes:
        unique_values = list(event_log[event_attribute].unique())
        att_array = np.array(unique_values)
        att_enc = OneHotEncoder(handle_unknown='ignore')
        att_enc.fit(att_array.reshape(-1, 1))
        attribute_encoder_list.append(att_enc)
        attribute_cardinality += len(unique_values)
    # List of one-hot encoders (for case attributes of intrest)
    case_encoder_list = []
    case_cardinality = 0
    for case_attribute in case_attributes_full:
        unique_values = list(event_log[case_attribute].unique())
        att_array = np.array(unique_values)
        att_enc = OneHotEncoder(handle_unknown='ignore')
        att_enc.fit(att_array.reshape(-1, 1))
        case_encoder_list.append(att_enc)
        case_cardinality += len(unique_values)           
    # Get node and edge dimensions
    node_dim = len(node_class_dict.keys()) # size for node featuers
    edge_dim  = attribute_cardinality + case_cardinality + case_num_card + event_num_card + 7 
    return (node_class_dict, max_case_df, max_active_cases, min_num_list,
            max_num_list, event_min_num_list, event_max_num_list,
            attribute_encoder_list, case_encoder_list, node_dim, edge_dim,
            avg_num_list, event_avg_num_list)