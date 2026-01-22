"""
@author: Keyvan Amiri Elyasi
"""
import yaml
import pm4py
import bisect
#import pickle

# Read user inputs from .yml file
def read_user_inputs(file_path):
    with open(file_path, 'r') as f:
        user_inputs = yaml.safe_load(f)
    return user_inputs

# Add string "case:" to categorical and numerical case attributes
def case_full_func(instance_att, instance_num_att):
    case_attributes_full, case_num_ful = [], []
    for case_attribute in instance_att:
        case_attribute = 'case:' + case_attribute
        case_attributes_full.append(case_attribute)
    for case_attribute in instance_num_att:
        case_attribute = 'case:' + case_attribute
        case_num_ful.append(case_attribute)        
    return case_attributes_full, case_num_ful

# Get train, validation, test data split, as well as duration of the longest case in training data
def train_test_split_func (log, event_log, split_ratio = [0.64, 0.16, 0.2]):
    start_dates, end_dates, durations, case_ids = [], [], [], []
    train_validation_index = int(len(log) * (split_ratio[0]+split_ratio[1]))
    train_index = int(len(log) * split_ratio[0])
    for i in range (len(log)):
        current_case = log[i]
        current_length = len(current_case)
        start_dates.append(current_case[0].get('time:timestamp'))
        end_dates.append(current_case[current_length-1].get('time:timestamp'))
        durations.append(
            (current_case[current_length-1].get('time:timestamp')
             - current_case[0].get('time:timestamp')).total_seconds()/3600/24)
        case_ids.append(current_case.attributes.get('concept:name'))        
    combined_data = list(zip(start_dates, end_dates, durations, case_ids))
    sorted_data = sorted(combined_data, key=lambda x: x[0])
    sorted_start_dates, sorted_end_dates, sorted_durations, sorted_case_ids = zip(*sorted_data)        
    train_case_ids = sorted_case_ids[:train_index]
    validation_case_ids = sorted_case_ids[train_index:train_validation_index]
    train_validation_case_ids = sorted_case_ids[:train_validation_index]
    test_case_ids = sorted_case_ids[train_validation_index:]
    train_validation_durations = sorted_durations[:train_validation_index]        
    max_case_duration = max(train_validation_durations)
    training_dataframe = pm4py.filter_trace_attribute_values(
        event_log, 'case:concept:name', train_case_ids, case_id_key='case:concept:name')
    validation_dataframe = pm4py.filter_trace_attribute_values(
        event_log, 'case:concept:name', validation_case_ids, case_id_key='case:concept:name')
    test_dataframe = pm4py.filter_trace_attribute_values(
        event_log, 'case:concept:name', test_case_ids, case_id_key='case:concept:name')
    train_validation_dataframe = pm4py.filter_trace_attribute_values(
        event_log, 'case:concept:name', train_validation_case_ids, case_id_key='case:concept:name')
    training_event_log = pm4py.convert_to_event_log(training_dataframe)
    validation_event_log = pm4py.convert_to_event_log(validation_dataframe)
    test_event_log = pm4py.convert_to_event_log(test_dataframe)
    train_validation_event_log = pm4py.convert_to_event_log(train_validation_dataframe)
    return (training_dataframe, validation_dataframe, test_dataframe,
            train_validation_dataframe,
            training_event_log, validation_event_log, test_event_log,
            train_validation_event_log,
            max_case_duration, sorted_start_dates, sorted_end_dates)

# Get the number of active cases (given a timestamp)
def ActiveCase(L1, L2, T):
    number_of_active_cases = bisect.bisect_right(L1, T) - bisect.bisect_right(L2, T)
    return number_of_active_cases

# Provide the class name for graph dataset (see: PGTNetlogHandler.py)
def eventlog_class_provider(name_of_dataset):
    if name_of_dataset == "BPIC15_1":
        pyg_class_name = "EVENTBPIC15M1"
    elif name_of_dataset == "BPIC15_2":
        pyg_class_name = "EVENTBPIC15M2"
    elif name_of_dataset == "BPIC15_3":
        pyg_class_name = "EVENTBPIC15M3"
    elif name_of_dataset == "BPIC15_4":
        pyg_class_name = "EVENTBPIC15M4"
    elif name_of_dataset == "BPIC15_5":
        pyg_class_name = "EVENTBPIC15M5"
    elif name_of_dataset == "BPI_Challenge_2012":
        pyg_class_name = "EVENTBPIC12"
    elif name_of_dataset == "BPI_Challenge_2012A":
        pyg_class_name = "EVENTBPIC12A"
    elif name_of_dataset == "BPI_Challenge_2012O":
        pyg_class_name = "EVENTBPIC12O"
    elif name_of_dataset == "BPI_Challenge_2012W":
        pyg_class_name = "EVENTBPIC12W"
    elif name_of_dataset == "BPI_Challenge_2012C":
        pyg_class_name = "EVENTBPIC12C"
    elif name_of_dataset == "BPI_Challenge_2012CW":
        pyg_class_name = "EVENTBPIC12CW"
    elif name_of_dataset == "BPI_Challenge_2013C" or name_of_dataset == "2013C":
        pyg_class_name = "EVENTBPIC13C"
    elif name_of_dataset == "BPI_Challenge_2013I" or name_of_dataset == "2013I":
        pyg_class_name = "EVENTBPIC13I"
    elif name_of_dataset == "BPIC20_DomesticDeclarations" or name_of_dataset == "2020D":
        pyg_class_name = "EVENTBPIC20D"
    elif name_of_dataset == "BPIC20_InternationalDeclarations" or name_of_dataset == "2020I":
        pyg_class_name = "EVENTBPIC20I"
    elif name_of_dataset == "env_permit" or name_of_dataset.lower() == "envpermit":
        pyg_class_name = "EVENTEnvPermit"
    elif name_of_dataset == "HelpDesk" or name_of_dataset.lower() == "helpdesk":
        pyg_class_name = "EVENTHelpDesk"
    elif name_of_dataset == "Hospital" or name_of_dataset.lower() == "hospital":
        pyg_class_name = "EVENTHospital"
    elif name_of_dataset == "Sepsis" or name_of_dataset.lower() == "sepsis":
        pyg_class_name = "EVENTSepsis"
    elif name_of_dataset == "Traffic_Fines" or name_of_dataset.lower() == "trafficfines":
        pyg_class_name = "EVENTTrafficfines" 
    else:
        pyg_class_name = None
        print('Error! no Pytorch Geometric dataset class is defined for this event log') 
    return pyg_class_name

"""
# Get subset of case IDs based on the Steady State defined in a dictionary
def get_subset_cases(ssd_dict, ssd_id, event_log, log, ssd_data_path):
    selected_cases = ssd_dict.get(ssd_id)
    subset_event_log = event_log[event_log['case:concept:name'].isin(selected_cases)]
    subset_log = pm4py.filter_trace_attribute_values(log, 'concept:name', selected_cases)
    pm4py.write_xes(subset_event_log, ssd_data_path)    
    return subset_event_log, subset_log
"""