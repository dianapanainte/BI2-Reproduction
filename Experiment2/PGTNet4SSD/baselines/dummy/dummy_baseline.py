# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 08:46:44 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import pm4py

from dummy_utils import add_rem_time, add_prefix_length, split_log
from dummy_utils import create_length_partitions
from dummy_utils import get_ssd_dict, ssd_bucket_dummy
from dummy_utils import no_bucket_dummy, length_bucket_dummy
from dummy_utils import get_clusters, clustering_bucket_dummy

def main():
    # TODO: adjust the data directory (in github repo)
    data_dir = r'C:\SNA-data\data'
    parser = argparse.ArgumentParser(
        description='DUMMY model for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--bucketing', type=str, default='C_B',
                        choices=['N_B', 'L_B', 'C_B', 'SSD_B'],
                        help='Bucketing Strategy to be used.')
    args = parser.parse_args()
    args.current_dir = os.getcwd()
    dataset_path = os.path.join(data_dir,args.dataset)
    log = pm4py.read_xes(dataset_path)
    df = add_rem_time(log)
    df = add_prefix_length(df)
    (train_case_ids, test_case_ids, train_df, test_df) = split_log(df)
    if args.bucketing == 'N_B':
        mae_test = no_bucket_dummy(train_df, test_df)
    elif args.bucketing == 'L_B':
        partitions = create_length_partitions(train_df)
        mae_test = length_bucket_dummy(train_df, test_df, partitions)
    elif args.bucketing == 'C_B':
        df_new, clusters = get_clusters(args, df, train_case_ids, test_case_ids) 
        mae_test = clustering_bucket_dummy(df_new, train_case_ids, test_case_ids, clusters)
    else:
        ssd_dict = get_ssd_dict(args)
        mae_test = ssd_bucket_dummy(train_df, test_df, ssd_dict)
    print(mae_test)    

if __name__ == '__main__':
    main()  