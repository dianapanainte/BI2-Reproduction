# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:04:43 2025
@author: Keyvan Amiri Elyasi
"""
import numpy as np
import torch
from torch_geometric.data import Data

from conversion_utils import ActiveCase

# main function for converting an event log into graph dataset
def graph_conversion_func(
        split_log, removed_cases, idx, data_list, case_attributes,
        case_encoder_list, case_num_att, min_num_list, max_num_list,
        event_attributes, event_num_att, target_normalization,
        max_time_norm, node_class_dict, edge_dim, max_case_df, 
        sorted_start_dates, sorted_end_dates, max_active_cases, 
        attribute_encoder_list, event_min_num_list, event_max_num_list,
        avg_num_list, event_avg_num_list):
    print("Check in the beginning function graph_conversion_func")
    # iterate over cases, and transform them if they have at least three events
    print("Total number of cases to process: " + str(len(split_log)))
    for case_counter in range(len(split_log)):
        print("Check - case_counter: " + str(case_counter))
        current_case = split_log[case_counter]
        case_id = split_log[case_counter].attributes.get('concept:name')   
        case_length = len(current_case)
        if case_length < 3:
            removed_cases.append(case_id)
        else:
            
            case_level_feat = np.empty((0,)) # collect all case-level information
            
            # first categorical attributes
            for att_index in range(len(case_attributes)):
                case_att = split_log[case_counter].attributes.get(case_attributes[att_index])
                case_att = str(case_att)
                #case_att_enc = case_encoder_list[att_index].transform(np.array(case_att).reshape(-1, 1)).toarray()
                case_att_enc = case_encoder_list[att_index].transform([[case_att]]).toarray()
                case_att_enc = case_att_enc.reshape(-1)
                case_level_feat = np.append(case_level_feat, case_att_enc)  
            
            # now, numerical attributes
            for att_index in range(len(case_num_att)):
                case_att = float(split_log[case_counter].attributes.get(case_num_att[att_index]))
                # impute NaN values with average value for that attribute!
                if np.isnan(case_att):
                    case_att_normalized = (avg_num_list[att_index] - min_num_list[att_index])/(max_num_list[att_index]- min_num_list[att_index])
                else:
                    case_att_normalized = (case_att - min_num_list[att_index])/(max_num_list[att_index]- min_num_list[att_index])
                case_level_feat = np.append(case_level_feat, np.array(case_att_normalized)) 
            
            # collect all events of the case, compute case start, end time
            case_events = split_log[case_counter][:]  
            case_start = split_log[case_counter][0].get('time:timestamp') 
            case_end = split_log[case_counter][case_length-1].get('time:timestamp')
        
            # collect activity classes, timestamps, and all attributes of intrest for each event
            collection_lists = [[] for _ in range(len(event_attributes)+len(event_num_att)+2)]
            for event_index in range(case_length):            
                current_event = case_events[event_index]
                collection_lists[0].append((current_event.get('concept:name'),current_event.get('lifecycle:transition')))
                collection_lists[1].append(current_event.get('time:timestamp'))
                for attribute_counter in range (2,len(event_attributes)+2):
                    collection_lists[attribute_counter].append(current_event.get(event_attributes[attribute_counter-2]))
                for attribute_counter in range (len(event_attributes)+2,len(event_attributes)+len(event_num_att)+2):
                    collection_lists[attribute_counter].append(current_event.get(event_num_att[attribute_counter-len(event_attributes)-2]))
        
            # for each prefix create a graph by iterating over all possible prefix lengthes
            for prefix_length in range (2, case_length):
                # prefix_event_classes is a list of tuples representing class = (act, life) of the relevant event.
                prefix_event_classes = collection_lists[0][:prefix_length]
                prefix_classes = list(set(prefix_event_classes)) # only includes unique classes
                prefix_times = collection_lists[1][:prefix_length]
            
                # create target based on the normalization option for user
                if target_normalization:
                    target_cycle = np.array((case_end - collection_lists[1][prefix_length-1]).total_seconds()/3600/24/max_time_norm)
                else:    
                    target_cycle = np.array((case_end - collection_lists[1][prefix_length-1]).total_seconds()/3600/24)
                y = torch.from_numpy(target_cycle).float()
                
                # collect information about nodes
                # define zero array to collect node features of the graph associated to this prefix  
                node_feature = np.zeros((len(prefix_classes), 1), dtype=np.int64)
                # collect node type by iteration over all nodes in the graph.
                for prefix_class in prefix_classes:
                    # get index of the relevant prefix class, and update its row in node feature matirx         
                    node_feature[prefix_classes.index(prefix_class)] = node_class_dict[prefix_class]
                x = torch.from_numpy(node_feature).long()
            
                # Compute edge index list.
                # Each item in pair_result: tuple of tuples representing df between two activity classes 
                pair_result = list(zip(prefix_event_classes , prefix_event_classes [1:]))
                pair_freq = {}            
                for item in pair_result:
                    source_index = prefix_classes.index(item[0])
                    target_index = prefix_classes.index(item[1])
                    if ((source_index, target_index) in pair_freq):
                        pair_freq[(source_index, target_index)] += 1
                    else:
                        pair_freq[(source_index, target_index)] = 1
                edges_list = list(pair_freq.keys())
                edge_index = torch.tensor(edges_list, dtype=torch.long)
            
                # Compute edge attributes
                edge_feature = np.zeros((len(edge_index), edge_dim), dtype=np.float64) # initialize edge feature matrix
                edge_counter = 0
                for edge in edge_index:
                    source_indices = [i for i, x in enumerate(prefix_event_classes) if x == prefix_classes[edge[0]]]
                    target_indices = [i for i, x in enumerate(prefix_event_classes) if x == prefix_classes[edge[1]]]
                    acceptable_indices = [(x, y) for x in source_indices for y in target_indices if x + 1 == y]
                    special_feat = np.empty((0,)) # collect all special features
                    # Add edge weights to the special feature vector
                    num_occ = len(acceptable_indices)/max_case_df
                    special_feat = np.append(special_feat, np.array(num_occ))
                    # Add temporal features to the special feature vector
                    sum_dur = 0
                    for acceptable_index in acceptable_indices:
                        last_dur = (prefix_times[acceptable_index[1]]- prefix_times[acceptable_index[0]]).total_seconds()/3600/24/max_time_norm
                        sum_dur += last_dur
                    special_feat = np.append(special_feat, np.array(last_dur))
                    special_feat = np.append(special_feat, np.array(sum_dur))
                    if acceptable_indices[-1][1] == prefix_length-1: # only meaningful for the latest event in prefix
                        temp_feat1 = (prefix_times[acceptable_indices[-1][1]]-case_start).total_seconds()/3600/24/max_time_norm
                        temp_feat2 = prefix_times[acceptable_indices[-1][1]].hour/24 + prefix_times[acceptable_indices[-1][1]].minute/60/24 + prefix_times[acceptable_indices[-1][1]].second/3600/24
                        temp_feat3 = (prefix_times[acceptable_indices[-1][1]].weekday() + temp_feat2)/7
                    else:
                        temp_feat1 = temp_feat2 = temp_feat3 = 0
                    special_feat = np.append(special_feat, np.array(temp_feat1))
                    special_feat = np.append(special_feat, np.array(temp_feat2))
                    special_feat = np.append(special_feat, np.array(temp_feat3))
                    num_cases = ActiveCase(sorted_start_dates, sorted_end_dates, prefix_times[acceptable_indices[-1][1]])/max_active_cases
                    special_feat = np.append(special_feat, np.array(num_cases))
                    partial_edge_feature = np.append(special_feat, case_level_feat)
                    
                    # One-hot encoding for the target of last occurence + numerical event attributes
                    for attribute_counter in range (2,len(event_attributes)+2):
                        attribute_value = np.array(collection_lists[attribute_counter][acceptable_indices[-1][1]]).reshape(-1, 1)
                        if str(attribute_value[0][0]) == 'nan':
                            num_zeros = len(attribute_encoder_list[attribute_counter - 2].categories_[0])
                            onehot_att = np.zeros((len(attribute_value), num_zeros))
                        else:
                            onehot_att = attribute_encoder_list[attribute_counter-2].transform(attribute_value).toarray()
                        partial_edge_feature = np.append(partial_edge_feature, onehot_att)
                    
                    # Numerical event attributes
                    # imputation requires improvement: average rather than using zero values!
                    for attribute_counter in range (len(event_attributes)+2,len(event_attributes)+len(event_num_att)+2):
                        attribute_value = np.array(collection_lists[attribute_counter][acceptable_indices[-1][1]])
                        if np.isnan(attribute_value):
                            norm_att_val = (event_avg_num_list[attribute_counter-len(event_attributes)-2] - event_min_num_list[attribute_counter-len(event_attributes)-2])/(event_max_num_list[attribute_counter-len(event_attributes)-2]- event_min_num_list[attribute_counter-len(event_attributes)-2])
                            #norm_att_val = np.array(0)
                        else:
                            norm_att_val = (attribute_value - event_min_num_list[attribute_counter-len(event_attributes)-2])/(event_max_num_list[attribute_counter-len(event_attributes)-2]- event_min_num_list[attribute_counter-len(event_attributes)-2])
                        partial_edge_feature = np.append(partial_edge_feature, norm_att_val)
                    edge_feature[edge_counter, :] = partial_edge_feature
                    edge_counter += 1
                edge_attr = torch.from_numpy(edge_feature).float()
                graph = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y, cid=case_id, pl = prefix_length)
                #print(graph)
                data_list.append(graph)
                idx += 1
    print("Check at the end of function graph_conversion_func")
    return removed_cases, idx, data_list