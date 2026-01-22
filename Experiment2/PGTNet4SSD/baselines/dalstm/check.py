# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 12:36:24 2025

@author: kamirel
"""
import os
import pickle

def main():
    directory = os.getcwd()
    name = 'DALSTM_train_case_list_HelpDesk.pkl'
    name = 'HelpDesk_mapping.pkl'
    check_path = os.path.join(directory,name)
    with open(check_path, 'rb') as file:
        check = pickle.load(file)
    print(check)

if __name__ == '__main__':
    main()