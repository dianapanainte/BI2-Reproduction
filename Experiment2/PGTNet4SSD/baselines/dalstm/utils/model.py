"""
@author: Keyvan Amiri Elyasi
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Data-aware LSTM model for remaining time prediction
class DALSTMModel(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, p_fix=0.2):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''
        super(DALSTMModel, self).__init__()
        
        self.n_layers = n_layers 
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        self.linear1 = nn.Linear(hidden_size, 1) 
        
    def forward(self, x):
        x = x.float() # if tensors are saved in a different format
        x, (hidden_state,cell_state) = self.lstm1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.batch_norm1(x)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                x, (hidden_state,cell_state) = self.lstm2(
                    x, (hidden_state,cell_state))
                if self.dropout:
                    x = self.dropout_layer(x)
                x = self.batch_norm1(x)
        yhat = self.linear1(x[:, -1, :]) # only the last one in the sequence 
        return yhat.squeeze(dim=1) 

# function to handle training the model
def train_model(model=None, train_loader=None, val_loader=None, criterion=None,
                optimizer=None, scheduler=None, device=None, 
                num_epochs=None, early_patience=None, min_delta=None,
                clip_grad_norm=None, clip_value=None,                 
                result_dir=None, bucket=None, bucket_idx=None, seed=None,
                logger=None):
    checkpoint_path = os.path.join(
        result_dir,
        'Bucket#{}#{}_seed_{}_model.pt'.format(bucket,bucket_idx,seed))     
    #Training loop
    start=datetime.now()
    current_patience = 0
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # training
        model.train()
        total_train_loss = 0.0  # reset every epoch
        for batch in train_loader:
            # Forward pass
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            optimizer.zero_grad() # Resets the gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            elif clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)    
            optimizer.step()  
            total_train_loss += loss.item()  # accumulate
        # compute average loss over all batches
        average_train_loss = total_train_loss / len(train_loader)
        # Validation
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            for batch in val_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(inputs)
                valid_loss = criterion(outputs, targets)
                total_valid_loss += valid_loss.item()                    
            average_valid_loss = total_valid_loss / len(val_loader)
        # print the results  
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss}, Validation Loss: {average_valid_loss}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss}, Validation Loss: {average_valid_loss}')
        # save the best model
        if average_valid_loss < best_valid_loss:
            best_valid_loss = average_valid_loss
            current_patience = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_valid_loss': best_valid_loss
                }
            torch.save(checkpoint, checkpoint_path)
        elif average_valid_loss - best_valid_loss >=  min_delta*best_valid_loss:
            current_patience += 1
            # Check for early stopping
            if current_patience >= early_patience:
                print('Early stopping: Val loss has not improved for {} epochs.'.format(early_patience))
                logger.info(f'Early stopping: Val loss has not improved for {early_patience} epochs')
                break  
        # Update learning rate if there is any scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(average_valid_loss)
            else:
                scheduler.step()
    # compute and report training time
    training_time = (datetime.now()-start).total_seconds()
    return training_time
           
# function to handle inference with trained model
def test_model(model=None, test_loader=None, test_original_lengths=None,
               y_scaler=None, normalization=False, device=None, 
               result_dir=None, bucket=None, bucket_idx=None, seed=None,
               logger=None):
    checkpoint_path = os.path.join(
        result_dir,
        'Bucket#{}#{}_seed_{}_model.pt'.format(bucket,bucket_idx,seed)) 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start=datetime.now()
    all_results = {'GroundTruth': [], 'Prediction': [], 'Prefix_length': [],
                   'Absolute_error': []}
    absolute_error = 0
    length_idx = 0 
    model.eval()
    with torch.no_grad():
        for index, test_batch in enumerate(test_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0]
            _y_pred = model(inputs)
            # convert tragets, outputs in case of normalization
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred        
            # Compute batch loss
            absolute_error += F.l1_loss(_y_pred, _y_truth).item()
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            mae_batch = np.abs(_y_truth - _y_pred)
            # collect inference result in all_result dict.
            all_results['GroundTruth'].extend(_y_truth.tolist())
            all_results['Prediction'].extend(_y_pred.tolist())
            pre_lengths = \
                test_original_lengths[length_idx:length_idx+batch_size]
            length_idx+=batch_size
            prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
            all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
    inference_time = (datetime.now()-start).total_seconds() 
    instance_inference = inference_time / len (test_original_lengths) * 1000
    print('Test - MAE: {:.3f}'.format(round(absolute_error, 3))) 
    logger.info(f'Results for bucket {bucket_idx},  Test - MAE (days): {absolute_error}')    
    flattened_list = [item for sublist in all_results['Prefix_length'] 
                      for item in sublist]
    all_results['Prefix_length'] = flattened_list
    results_df = pd.DataFrame(all_results)
    results_df['bucket'] = bucket_idx
    return results_df, instance_inference
    
