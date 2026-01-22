import os
import sys
import re
import argparse
#import datetime
import logging
import pickle
import pandas as pd 
import numpy as np 
import torch
import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

from PGTNet4SSD.PGTNetutils import get_ssd_name, get_length_buckets
#from PGTNet4SSD.PGTNetutils import mean_cycle_norm_factor_provider, eventlog_name_provider 


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def get_info_from_logger(log_file):
    num_graphs = None
    test_mae = None
    with open(log_file, "r") as f:
        for line in f:
            # Search for number of graphs
            if "num graphs" in line.lower():
                match = re.search(r"\b\d+\b", line)
                if match:
                    num_graphs = int(match.group(0))        
            # Search for test_mae
            if "test_mae" in line.lower():
                match = re.search(r"test_mae\s*[:=]\s*([0-9.]+)", line)
                if match:
                    test_mae = float(match.group(1))
    return num_graphs, test_mae

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    #cfg.out_dir = os.path.join(cfg.out_dir, run_name, cfg.bucketing, cfg.segment_idx)
    cfg.out_dir = os.path.join(cfg.result_directory, run_name, cfg.bucketing, cfg.segment_idx)
    

def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

def run_loop_settings():
    """Create main loop execution settings based on the current cfg.
    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.
    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = list(range(num_iterations))
        #run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    #print(run_ids)
    #print(seeds)
    #print(split_indices)
    return run_ids, seeds, split_indices

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucketing', type=str, default='SSD_B', 
                        choices=['N_B', 'L_B', 'C_B', 'SSD_B'],
                        help='Bucketing Strategy to be used.')
    custom_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    # Load cmd line args
    args = parse_args()  
    root_dir = os.getcwd()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    cfg.bucketing = custom_args.bucketing
    cfg.result_directory = os.path.join(root_dir, 'results')
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # load auxiliary data    
    dataset_name = cfg.dataset.format.split("PyG-")[1]
    dataset_folder = os.path.join(root_dir, 'datasets', dataset_name)    
    norm_path = os.path.join(dataset_folder, 'max_time_norm.pkl')  
    with open(norm_path, "rb") as f:
        norm_constant = pickle.load(f)
    if cfg.bucketing == 'N_B':
        segment_indices = [0]
    elif cfg.bucketing == 'L_B':
        length_buckets = get_length_buckets(dataset_name)
        segment_indices = list(range(len(length_buckets)))
    elif cfg.bucketing == 'C_B':
        cluster_path = os.path.join(dataset_folder, 'test_clusters.pkl')
        with open(cluster_path, "rb") as f:
            test_clusters = pickle.load(f) 
        segment_indices = list(test_clusters.keys())
    else:
        SSD_name = get_ssd_name(dataset_name)
        SSD_path = os.path.join(root_dir, 'datasets', 'SSD_results', SSD_name)
        with open(SSD_path, "rb") as f:
            SSD_dict = pickle.load(f) 
        if dataset_name == "EVENTHelpDesk":
            SSD_dict[1].extend(SSD_dict[2])
            del SSD_dict[2]
        segment_indices = list(SSD_dict.keys())

    mae_dict = {'segments': [], 'seeds': [], 'num_samples': [], 'MAE': []}
    
    for segment_idx, segment in enumerate(segment_indices):
        cfg.segment_idx = str(segment_idx)
        custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
        dump_cfg(cfg)
        if cfg.bucketing == 'C_B':
            cfg.segment = segment
        elif cfg.bucketing == 'SSD_B':
            cfg.segment = SSD_dict[segment]
        elif cfg.bucketing == 'L_B':
            cfg.segment = length_buckets[segment_idx]
        # Repeat for multiple experiment runs    
        for run_id, seed, split_index in zip(*run_loop_settings()):
            # Set configurations for each run
            custom_set_run_dir(cfg, run_id)
            set_printing()
            cfg.dataset.split_index = split_index
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            auto_select_device()
            logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                         f"split_index={cfg.dataset.split_index}")
            # Set machine learning pipeline
            loaders = create_loader()
            print('##################')
            print(len(loaders[0]), len(loaders[1]), len(loaders[2]))
            #print(len(loaders[0]), len(loaders[1]), len(loaders[2]))
            loggers = create_logger()
            if len(loaders[2]) == 0:
                print('There is no sample for this bucket in test set')
                continue
            model = create_model()
            optimizer = create_optimizer(model.parameters(),
                                         new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            # Print model info
            logging.info(model)
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info('Num parameters: %s', cfg.params)
            # Start training
            if cfg.train.mode == 'standard':
                if cfg.wandb.use:
                    logging.warning("[W] WandB logging is not supported with the "
                                    "default train.mode, set it to `custom`")
                datamodule = GraphGymDataModule()
                train(model, datamodule, logger=True)
            else:
                train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                               scheduler)
            # collect the MAE information
            logger_file_path = os.path.join(cfg.run_dir, 'logging.log')
            num_graphs, test_mae = get_info_from_logger(logger_file_path)
            mae_dict['segments'].append(segment_idx)
            mae_dict['seeds'].append(seed)
            mae_dict['num_samples'].append(num_graphs)
            mae_dict['MAE'].append(test_mae*norm_constant)
    mae_df = pd.DataFrame(mae_dict)
    run_name = os.path.splitext(os.path.basename(args.cfg_file))[0]
    run_name += f"-{cfg.name_tag}" if cfg.name_tag else ""
    csv_name = dataset_name + '_' + cfg.bucketing + '_MAE_Report.csv'
    csv_path = os.path.join(cfg.result_directory, run_name, csv_name)
    mae_df.to_csv(csv_path, index=True)
    # group by num_samples
    groups = {ns: g.reset_index(drop=True) 
              for ns, g in mae_df.groupby('num_samples')}
    num_subsets = min(len(g) for g in groups.values())
    weighted_mae_list = []
    for i in range(num_subsets):
        subset = pd.concat([g.iloc[[i]] for g in groups.values()])
        # weighted MAE using num_samples as weights
        weighted_mae = (subset['MAE'] * subset['num_samples']).sum() / subset['num_samples'].sum()
        weighted_mae_list.append(weighted_mae)
    mean_mae = np.mean(weighted_mae_list)
    std_mae  = np.std(weighted_mae_list)
    rep_name = dataset_name + '_MAE_Report.txt'
    rep_path = os.path.join(cfg.result_directory, run_name, rep_name)
    if os.path.exists(rep_path):
        with open(rep_path, "a") as f:
            f.write(f"For bucketing: {cfg.bucketing}: mean MAE: {mean_mae}, std MAE {std_mae}.\n")
    else:
        with open(rep_path, "w") as f:
            f.write(f"For bucketing: {cfg.bucketing}: mean MAE: {mean_mae}, std MAE {std_mae}.\n")

    
    

        

    """
        if cfg.train.mode == 'standard':
            #### something!
        # Achtung! the following is added to the original implementation of GraphGPS framework
        elif cfg.train.mode == 'event-inference':
            fold_address = cfg.pretrained.dir + f"/{run_id}/ckpt"
            # load the saved check point (Parameters with lowest validation loss)
            if os.path.exists(fold_address) and os.path.isdir(fold_address):
                check_point_files = os.listdir(fold_address)
                if len(check_point_files) == 1:
                    check_point_name = os.path.basename(check_point_files[0])
                    checkpoint_path = os.path.join(fold_address, check_point_name)
                    loaded_checkpoint = torch.load(checkpoint_path)
                    model_state_dict = loaded_checkpoint['model_state']
                else:
                    print("Error: multiple check point files might have been found.")
            else:
                print("Error: the specified folder does not exist.")
            # update the model parameters by those obtained from check point file.
            model.load_state_dict(model_state_dict)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set the device
            model.to(device) # move the model
            model.eval()  # Set the model to evaluation mode
            # set empty lists to create the final dataframe for each fold (run)
            num_node_list, num_edge_list, real_reamining_times, predictions = [], [], [], []
            evaluation_test_loader = loaders[2] # use only test set as the loader
            with torch.no_grad():
                for each_graph in evaluation_test_loader:                    
                    each_graph.to(device) # move the test example to device
                    graph_transformer_prediction = model(each_graph) # get prediction of the model
                    predictions.append(float(np.array(graph_transformer_prediction[0].cpu()))) # tuple of value & device
                    num_node_list.append(each_graph.x.shape[0])  # get number of nodes                  
                    num_edge_list.append(each_graph.edge_attr.shape[0]) # get number of edges
                    real_reamining_times.append(np.array(each_graph.y[0].cpu())) # get real reamining time
            Aggregated_graph_info = {'num_node': num_node_list, 'num_edge': num_edge_list,
                                     'real_cycle_time': real_reamining_times,
                                     'predicted_cycle_time': predictions}
            evalauation_test_dataframe = pd.DataFrame(Aggregated_graph_info) # convert to dataframe
            inference_dataframes.append(evalauation_test_dataframe)


    # Aggregate results from different seeds
    # Achtung! the following CONDITION is added to the original implementation of GraphGPS framework.
    # in the original implementation try-except runs without any CONDITION
    if cfg.train.mode != 'event-inference':
        try:
            agg_runs(cfg.out_dir, cfg.metric_best)
        except Exception as e:
            logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # Achtung! the following is added to aggregate the result in event-inference mode
    if cfg.train.mode == 'event-inference':
        prediction_dataframe = pd.concat(inference_dataframes, ignore_index=True)
        dataset_class_name = cfg.dataset.format.split('-')[1]
        event_log_name = eventlog_name_provider(dataset_class_name)
        prediction_file_name = event_log_name + '-pgtnet_prediction_dataframe.csv'
        if cfg.ssd:
            root_path = os.getcwd()
            load_path= os.path.join(
                root_path, 'PGTNet4SSD', 'raw_dataset', 
                event_log_name + '#normalization_factor.pkl')
            normalization_factor, mean_cycle = mean_cycle_norm_factor_provider(
                event_log_name, load_path=load_path, ssd=True)
        else:
            normalization_factor, mean_cycle = mean_cycle_norm_factor_provider(
                dataset_class_name)
        prediction_dataframe['MAE-days'] = (prediction_dataframe['real_cycle_time'] - prediction_dataframe['predicted_cycle_time']).abs() * normalization_factor
        evalauation_df_path = os.path.join(cfg.out_dir, prediction_file_name)
        prediction_dataframe.to_csv(evalauation_df_path, index=False) # save prediction dataframe
        inference_end_time = datetime.datetime.now()
        inference_time = (inference_end_time - inference_start_time).total_seconds() * 1000
        inference_time_per_prefix = inference_time/(len(loaders[0])+len(loaders[1])+len(loaders[2]))
        mean_absolute_error = prediction_dataframe['MAE-days'].mean()
        print('MAE (days):',  mean_absolute_error)
        relative_mean_absolute_error = mean_absolute_error/mean_cycle*100
        print('Relative MAE (days) for dataset:',  relative_mean_absolute_error)
        print('Inference time per event prefix:',  inference_time_per_prefix)
    # now, back to the original implementation
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
    """
    
    
    
    """
        # Achtung! the following is added to keep track of inference time
        if cfg.train.mode == 'event-inference' and run_id == 0:
            inference_start_time = datetime.datetime.now()
        if cfg.train.mode == 'event-inference' and cfg.ssd:
            inference_start_time = datetime.datetime.now()
        logging.info(f"    Starting now: {datetime.datetime.now()}")
    """