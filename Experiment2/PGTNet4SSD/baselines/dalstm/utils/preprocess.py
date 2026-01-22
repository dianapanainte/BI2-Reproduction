"""
This script is based on the following source code:
    https://gitlab.citius.usc.es/efren.rama/pmdlcompararator
We just adjusted some parts to efficiently use it in our study.
@author: Keyvan Amiri Elyasi
"""
import os
import pickle
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pm4py
import torch
from pm4py.objects.log.importer.xes import importer as xes_import_factory
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence

from utils.utils import get_csv_paths, get_tensor_paths, get_length_paths
from utils.utils import get_id_paths, get_cluster_paths, get_shape_paths
from utils.utils import delete_files
from utils.bucket import create_clusters, assign_clusters 

##############################################################################
# Data preprocessing utility methods, and classes
##############################################################################   

class XES_Fields:
    CASE_COLUMN = 'case:concept:name'
    ACTIVITY_COLUMN = 'concept:name'
    TIMESTAMP_COLUMN = 'time:timestamp'
    LIFECYCLE_COLUMN = 'lifecycle:transition'
    
class Timestamp_Formats:
    TIME_FORMAT_DALSTM = "%Y-%m-%d %H:%M:%S"
    #TIME_FORMAT_DALSTM2 = '%Y-%m-%d %H:%M:%S.%f%z' 
    TIME_FORMAT_DALSTM2 = '%Y-%m-%d %H:%M:%S%z' # all BPIC 2012 logs
    TIME_FORMAT_DALSTM_list = [TIME_FORMAT_DALSTM, TIME_FORMAT_DALSTM2]
    
def buildOHE(index, n):
    L = [0] * n
    L[index] = 1
    return L

# A method to tranform XES to CSV and execute some preprocessing steps
def xes_to_csv(file, output_folder, perform_lifecycle_trick=True, fill_na=None):    
    xes_path = file
    csv_file = Path(file).stem.split('.')[0] + '.csv'
    dataset_name = Path(file).stem.split('.')[0]
    csv_path = os.path.join(output_folder, csv_file)
    case_id_map_name = dataset_name + '_mapping.pkl'
    case_id_map_path = os.path.join(output_folder, case_id_map_name)
    log = xes_import_factory.apply(xes_path, parameters={'timestamp_sort': True})   
    equivalent_dataframe = pm4py.convert_to_dataframe(log)
    equivalent_dataframe.to_csv(csv_path)
    pd_log = pd.read_csv(csv_path)   
    if fill_na is not None:
        pd_log.fillna(fill_na, inplace=True)
        pd_log.replace("-", fill_na, inplace=True)
        pd_log.replace(np.nan, fill_na)
    if 'BPI_Challenge_2012' in dataset_name:
        counter_list = []
        for counter in range (len(pd_log)):
            for format_str in Timestamp_Formats.TIME_FORMAT_DALSTM_list:
                try:
                    incr_timestamp = datetime.strptime(
                        str(pd_log.iloc[counter][
                            XES_Fields.TIMESTAMP_COLUMN]), format_str)  
                    if format_str == '%Y-%m-%d %H:%M:%S%z':
                        counter_list.append(counter)
                    break
                except ValueError:
                    continue
        pd_log = pd_log.drop(index=counter_list)
    # Use integers always for case identifiers.
    # We need this to make a split that is equal for every dataset
    pd_log[XES_Fields.CASE_COLUMN] = pd.Categorical(
        pd_log[XES_Fields.CASE_COLUMN])
    # keep track of original case IDs
    case_id_map = dict(enumerate(pd_log[XES_Fields.CASE_COLUMN].cat.categories))
    #reverse_case_id_map = {v: k for k, v in case_id_map.items()}
    pd_log[XES_Fields.CASE_COLUMN] = pd_log[XES_Fields.CASE_COLUMN].cat.codes 
    # lifecycle_trick: ACTIVITY NAME + LIFECYCLE-TRANSITION
    try:
        unique_lifecycle = pd_log[XES_Fields.LIFECYCLE_COLUMN].unique()
    except:
        # to handle situations in which there is no lifecycle information
        unique_lifecycle = ['COMPLETE'] 
    if len(unique_lifecycle) > 1 and perform_lifecycle_trick:
        pd_log[XES_Fields.ACTIVITY_COLUMN] = pd_log[
            XES_Fields.ACTIVITY_COLUMN].astype(str) + "+" + pd_log[
                XES_Fields.LIFECYCLE_COLUMN]       
    pd_log.to_csv(csv_path, encoding="utf-8")
    with open(case_id_map_path, 'wb') as file:
        pickle.dump(case_id_map, file)
    return csv_file, csv_path

def select_columns(file, input_columns, category_columns, timestamp_format,
                   output_columns, categorize=False, fill_na=None,
                   save_category_assignment=None):

    dataset = pd.read_csv(file)
    if fill_na is not None:
        dataset = dataset.fillna(fill_na)
    if input_columns is not None:
        dataset = dataset[input_columns]
    timestamp_column = XES_Fields.TIMESTAMP_COLUMN
    dataset[timestamp_column] = pd.to_datetime(
        dataset[timestamp_column], utc=True)
    dataset[timestamp_column] = dataset[
        timestamp_column].dt.strftime(timestamp_format)
    if categorize:
        for category_column in category_columns:
            if category_column == XES_Fields.ACTIVITY_COLUMN:
                category_list = dataset[
                    category_column].astype("category").cat.categories.tolist()
                category_dict = {c : i for i, c in enumerate(category_list)}
                if save_category_assignment is None:
                    print("Activity assignment: ", category_dict)
                else:
                    file_name = Path(file).name
                    with open(os.path.join(
                            save_category_assignment, file_name), "w") as fw:
                        fw.write(str(category_dict))
            dataset[category_column] = dataset[
                category_column].astype("category").cat.codes
    if output_columns is not None:
        dataset.rename(
            output_columns,
            axis="columns",
            inplace=True)
    dataset.to_csv(file, sep=",", index=False)
    
def reorder_columns(file, ordered_columns):
    df = pd.read_csv(file)
    df = df.reindex(columns=(ordered_columns + list(
        [a for a in df.columns if a not in ordered_columns])))
    df.to_csv(file, sep=",", index=False)

# A method to split the cases into training, validation, and test sets
def split_data(file=None, output_directory=None, case_column=None,
               train_val_test_ratio = [0.64, 0.16, 0.2]):
    # split data for cv
    pandas_init = pd.read_csv(file)
    pd.set_option('display.expand_frame_repr', False)
    groups = [pandas_df for _, pandas_df in \
              pandas_init.groupby(case_column, sort=False)]
    train_size = round(len(groups) * train_val_test_ratio[0])
    val_size = round(len(groups) * (train_val_test_ratio[0]+\
                                    train_val_test_ratio[1]))
    train_groups = groups[:train_size]
    val_groups = groups[train_size:val_size]
    test_groups = groups[val_size:]
    # Disable the sorting. Otherwise it would mess with the order of the timestamps
    train = pd.concat(train_groups, sort=False).reset_index(drop=True)
    val = pd.concat(val_groups, sort=False).reset_index(drop=True)
    test = pd.concat(test_groups, sort=False).reset_index(drop=True)
    train_hold_path = os.path.join(output_directory, "train_" + Path(file).stem + ".csv")
    val_hold_path = os.path.join(output_directory, "val_" + Path(file).stem + ".csv")
    test_hold_path = os.path.join(output_directory, "test_" + Path(file).stem + ".csv")
    train.to_csv(train_hold_path, index=False)
    val.to_csv(val_hold_path, index=False)
    test.to_csv(test_hold_path, index=False)
    
# method to handle initial steps of preprocessing for DALSTM
def data_handling(xes=None, output_folder=None, cfg=None):
    dataset_name = Path(xes).stem.split('.')[0] 
    csv_file, csv_path = xes_to_csv(file=xes, output_folder=output_folder) 
    # Define relevant attributes
    attributes = cfg[dataset_name]['event_attributes']
    if dataset_name == "Traffic_Fines":
        attributes.remove('dismissal')     
    if (dataset_name == "BPI_Challenge_2012" or dataset_name == "BPI_2012W" 
        or dataset_name == "BPI_2013_I"):
        attributes.append(XES_Fields.LIFECYCLE_COLUMN)
    # select related columns
    if 'BPI_Challenge_2012' in dataset_name:
        #selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM2
        selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM
    else:
        selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM
    select_columns(csv_path, input_columns=[XES_Fields.CASE_COLUMN,
                                            XES_Fields.ACTIVITY_COLUMN,
                                            XES_Fields.TIMESTAMP_COLUMN] + attributes,
                   category_columns=None, 
                   timestamp_format=selected_timestamp_format,
                   output_columns=None, categorize=False) 
    # Reorder columns   
    reorder_columns(csv_path, [XES_Fields.CASE_COLUMN,
                               XES_Fields.ACTIVITY_COLUMN,
                               XES_Fields.TIMESTAMP_COLUMN])  
    # execute data split
    split_data(file=csv_path, output_directory=output_folder,
               case_column=XES_Fields.CASE_COLUMN)

    
# A method for DALSTM preprocessing (output: Pytorch tensors for training)
def dalstm_load_dataset(filename, prev_values=None):
    #dataset_name = os.path.splitext(os.path.basename(filename))[0]
    dataframe = pd.read_csv(filename, header=0)
    dataframe = dataframe.replace(r's+', 'empty', regex=True)
    dataframe = dataframe.replace("-", "UNK")
    dataframe = dataframe.fillna(0)

    dataset = dataframe.values
    if prev_values is None:
        values = []
        for i in range(dataset.shape[1]):
            try:
                values.append(len(np.unique(dataset[:, i])))  # +1
            except:
                dataset[:, i] = dataset[:, i].astype(str)       
                values.append(len(np.unique(dataset[:, i])))  # +1

        # output is changed to handle prefix lengths
        #print(values)
        return (None, None, None, None), values 
    else:
        values = prev_values

    #print("Dataset: ", dataset)
    #print("Values: ", values)

    datasetTR = dataset

    def generate_set(dataset):
        # To collect case IDs (required for SSD based bucketing)
        all_ids = dataset[:, 0].astype(str).tolist()        
        counts = Counter(all_ids)
        case_ids = []
        for item in all_ids:
            if counts[item] > 0:
                counts[item] -= 1
                if counts[item] >= 2:
                    case_ids.append(item)

        data = []
        # To collect prefix lengths (required for length based bucketing)
        original_lengths = [] 
        newdataset = []
        temptarget = []
        
        # analyze first dataset line
        caseID = dataset[0][0]
        starttime = datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        lastevtime = datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        t = time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
        midnight = datetime.fromtimestamp(
            time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = (
            datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
        n = 1
        temptarget.append(
            datetime.fromtimestamp(time.mktime(
                time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))))
        a = [(datetime.fromtimestamp(
            time.mktime(time.strptime(
                dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
        a.append((datetime.fromtimestamp(
            time.mktime(time.strptime(
                dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
        a.append(timesincemidnight)
        a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
        a.extend(buildOHE(
            one_hot(dataset[0][1], values[1], split="|")[0], values[1]))

        field = 3
        for i in dataset[0][3:]:
            if not np.issubdtype(dataframe.dtypes[field], np.number):
                #print(field, values[field])           
                a.extend(buildOHE(one_hot(
                    str(i), values[field], split="|")[0], values[field]))
            else:
                #print('numerical', field)
                a.append(i)
            field += 1
        newdataset.append(a)
        #line_counter = 1
        for line in dataset[1:, :]:
            #print(line_counter)
            case = line[0]
            if case == caseID:
                # continues the current case
                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(
                    time.mktime(t)) - midnight).total_seconds()
                temptarget.append(datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                a = [(datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                a.extend(buildOHE(one_hot(
                    line[1], values[1], filters=[], split="|")[0], values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(
                            dataframe.dtypes[field], np.number):
                        a.extend(buildOHE(
                            one_hot(str(i), values[field], filters=[],
                                    split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                n += 1
                finishtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            else:
                caseID = case
                # Exclude prefix of length one: the loop range is changed.
                # +1 not adding last case. target is 0, not interesting. era 1
                for i in range(2, len(newdataset)): 
                    data.append(newdataset[:i])
                    # Keep track of prefix lengths (earliness analysis)
                    original_lengths.append(i) 
                    # print newdataset[:i]
                newdataset = []
                starttime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(
                    time.mktime(t)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (
                    datetime.fromtimestamp(
                        time.mktime(t)) - midnight).total_seconds()

                a = [(datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(
                        line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                a.extend(buildOHE(one_hot(line[1], values[1], split="|")[0], values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        a.extend(buildOHE(
                            one_hot(str(i), values[field],
                                    split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                for i in range(n):  
                    # try-except: error handling of the original implementation.
                    try:
                        temptarget[-(i + 1)] = (
                            finishtime - temptarget[-(i + 1)]).total_seconds()
                    except UnboundLocalError:
                        # Set target value to zero if finishtime is not defined
                        # The effect is negligible as only for one dataset,
                        # this exception is for one time executed
                        print('one error in loading dataset is observed', i, n)
                        temptarget[-(i + 1)] = 0
                # Remove the target attribute for the prefix of length one
                if n > 1:
                    temptarget.pop(0-n)
                temptarget.pop()  # remove last element with zero target
                temptarget.append(
                    datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], "%Y-%m-%d %H:%M:%S"))))
                finishtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                n = 1
            #line_counter += 1
        # last case
        # To exclude prefix of length 1: the loop range is adjusted.
        # + 1 not adding last event, target is 0 in that case. era 1
        for i in range(2, len(newdataset)):  
            data.append(newdataset[:i])
            original_lengths.append(i) # Keep track of prefix lengths
            # print newdataset[:i]
        for i in range(n):  # era n.
            temptarget[-(i + 1)] = (
                finishtime - temptarget[-(i + 1)]).total_seconds()
            # print temptarget[-(i + 1)]
        # Remove the target attribute for the prefix of length one
        if n > 1:
            temptarget.pop(0-n)
        temptarget.pop()  # remove last element with zero target

        # print temptarget
        print("Generated dataset with n_samples:", len(temptarget))
        assert (len(temptarget) == len(data))
        #Achtung! original_lengths is added to output
        return data, temptarget, original_lengths, case_ids

    return generate_set(datasetTR), values

# A method for DALSTM preprocessing (prepoessing actions on pytorch tensors)
def dalstm_process(dataset_name=None, output_folder=None, normalization=False,
                   cv_split=False, n_splits=5, delete_csv_files=False):
    # define important file names and paths
    (full_dataset_path,train_dataset_path,
     val_dataset_path,test_dataset_path) = get_csv_paths(
         dataset_name, output_folder)
    # call dalstm_load_dataset for the whole dataset
    (_, _, _,_), values = dalstm_load_dataset(full_dataset_path)
    # call dalstm_load_dataset for training, validation, and test sets
    (X_train, y_train, train_lengths, train_ids), _ =  dalstm_load_dataset(
        train_dataset_path, values)
    (X_val, y_val, valid_lengths, val_ids), _ = dalstm_load_dataset(
        val_dataset_path, values)
    (X_test, y_test, test_lengths, test_ids), _ = dalstm_load_dataset(
        test_dataset_path, values) 
    # normalize input data
    # compute the normalization values only on training set
    max = [0] * len(X_train[0][0])
    for a1 in X_train:
        for s in a1:
            for i in range(len(s)):
                if s[i] > max[i]:
                    max[i] = s[i]
    # normalization for train, validation, and test sets
    for a1 in X_train:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    for a1 in X_val:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    for a1 in X_test:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]    
    # convert the results to numpy arrays
    X_train = np.asarray(X_train, dtype='object')
    X_val = np.asarray(X_val, dtype='object')
    X_test = np.asarray(X_test, dtype='object')
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    # execute padding, and error handling for BPIC13I
    if dataset_name == 'BPI_2013_I':
        X_train = sequence.pad_sequences(X_train, dtype="int16")
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], 
                                        dtype="int16")
        X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1],
                                       dtype="int16")
    else:
        X_train = sequence.pad_sequences(X_train)
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
        X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1])
    # Convert target attribute to days
    y_train /= (24*3600) 
    y_val /= (24*3600) 
    y_test /= (24*3600) 
    # Target attribute normalization
    if normalization:
        max_y_train = np.max(y_train)
        max_y_val = np.max(y_val)
        max_train_val = np.max([max_y_train, max_y_val])
        #print(max_train_val)
        y_train /= max_train_val
        y_val /= max_train_val
        y_test /= max_train_val
    else:
        max_train_val = None
    # apply decision based clustering for bucketing
    cluster_model = create_clusters(X_train, X_val, y_train, y_val)
    clusters_train, clusters_val, clusters_test = assign_clusters(
        cluster_model, X_train, X_val, X_test)       
    # convert numpy arrays to tensors
    # manage disk space for huge event logs
    if (('BPIC15' in dataset_name) or (dataset_name== 'Traffic_Fines') or
        (dataset_name== 'Hospital')):
        X_train = torch.tensor(X_train).type(torch.bfloat16)
        X_val = torch.tensor(X_val).type(torch.bfloat16)
        X_test = torch.tensor(X_test).type(torch.bfloat16)
    else:
        X_train = torch.tensor(X_train).type(torch.float)
        X_val = torch.tensor(X_val).type(torch.float)
        X_test = torch.tensor(X_test).type(torch.float)
    y_train = torch.tensor(y_train).type(torch.float)
    y_val = torch.tensor(y_val).type(torch.float)
    y_test = torch.tensor(y_test).type(torch.float)
    input_size = X_train.size(2)
    max_len = X_train.size(1) 
    # save tensor data
    (X_train_path,X_val_path,X_test_path,y_train_path,y_val_path,y_test_path
     ) = get_tensor_paths(dataset_name, output_folder)
    torch.save(X_train, X_train_path)                    
    torch.save(X_val, X_val_path)
    torch.save(X_test, X_test_path)                      
    torch.save(y_train, y_train_path)
    torch.save(y_val, y_val_path)
    torch.save(y_test, y_test_path)
    # save lengths
    (train_length_path,val_length_path,test_length_path) = get_length_paths(
        dataset_name, output_folder)
    with open(train_length_path, 'wb') as file:
        pickle.dump(train_lengths, file)
    with open(val_length_path, 'wb') as file:
        pickle.dump(valid_lengths, file)
    with open(test_length_path, 'wb') as file:
        pickle.dump(test_lengths, file)
    # save case IDs
    (train_id_path,val_id_path,test_id_path) = get_id_paths(
        dataset_name, output_folder)
    with open(train_id_path, 'wb') as file:
        pickle.dump(train_ids, file)
    with open(val_id_path, 'wb') as file:
        pickle.dump(val_ids, file)
    with open(test_id_path, 'wb') as file:
        pickle.dump(test_ids, file)
    # save clusters
    (train_cluster_path,val_cluster_path,test_cluster_path) = get_cluster_paths(
        dataset_name, output_folder)
    with open(train_cluster_path, 'wb') as file:
        pickle.dump(clusters_train, file)
    with open(val_cluster_path, 'wb') as file:
        pickle.dump(clusters_val, file)
    with open(test_cluster_path, 'wb') as file:
        pickle.dump(clusters_test, file)
    # save auxiliary information
    (scaler_path, input_size_path, max_len_path) = get_shape_paths(
        dataset_name, output_folder)
    with open(scaler_path, 'wb') as file:
        pickle.dump(max_train_val, file)
    with open(input_size_path, 'wb') as file:
        pickle.dump(input_size, file)
    with open(max_len_path, 'wb') as file:
        pickle.dump(max_len, file)
    # Delete csv files as they are not require anymore
    if delete_csv_files:
        delete_files(folder_path=output_folder, extension='.csv')
    print('Preprocessing is done for holdout data split.')