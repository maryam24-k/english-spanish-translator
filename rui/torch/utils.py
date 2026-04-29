#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of utility functions for CIS433
@author: huaxia
"""
 
import os
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
 
#%%
 
class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", save_best_only=True, save_optimizer_state=False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_optimizer_state = save_optimizer_state
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
 
    def on_best(self, model, optimizer):
        if self.save_optimizer_state:
            torch.save( {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, self.filepath)
        else:
            torch.save(model.state_dict(), self.filepath)
 
 
def plotEpoch(history, metric="average_loss"):
    epochs = range(len(history["average_train_loss"]))
    if metric == "accuracy":
        plt.plot(epochs, history["train_accuracy"], "k", label="Training acc")
        plt.plot(epochs, history["val_accuracy"], "b", label="Validation acc")
        plt.axvline( x=int(np.argmax(history["val_accuracy"])), color="0.5" )
        plt.title("Training and Validation Accuracy")
        plt.legend()
    elif metric == "total_loss":
        plt.plot(epochs, history["total_train_loss"], "k", label="Training loss")
        plt.plot(epochs, history["total_val_loss"], "b", label="Validation loss")
        plt.axvline(x=int(np.argmin(history["total_val_loss"])), color="0.5")
        plt.title("Total Training and Validation Loss")
        plt.legend()
    elif metric == "average_loss":
        plt.plot(epochs, history["average_train_loss"], "k", label="Training loss")
        plt.plot(epochs, history["average_val_loss"], "b", label="Validation loss")
        plt.axvline(x=int(np.argmin(history["average_val_loss"])), color="0.5")
        plt.title("Average Training and Validation Loss")
        plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()    
 
 
def stats_per_channel(loader):
    n_channels, _, _ = loader.dataset[0][0].size() # assuming each batch is a tuple of (x,y)
    #total_samples = torch.ones( (1, n_channels) ) * len(loader.dataset)
    total_means = torch.zeros( (1,n_channels) )
    total_stds = torch.zeros( (1,n_channels) )
    for images, _ in loader:        
        flatten_per_channel = torch.flatten(images, start_dim=-2, end_dim=-1) # (N,C,H*W)
        means = flatten_per_channel.mean(axis=2) # (N,C)
        stds = flatten_per_channel.std(axis=2) # (N,C)
        total_means += means.sum(axis=0) # (1, n_channels)
        total_stds += stds.sum(axis=0) # (1, n_channels)
    norm_mean = total_means / len(loader.dataset) # (1, n_channels)
    norm_std = total_stds / len(loader.dataset) # (1, n_channels)
    return norm_mean.squeeze(), norm_std.squeeze() # without squeezing, runtime error: not all boolean value of tensor with more than one value is ambiguous
 
#%% Train Loop: tested on classification, regression, image segmentation, translation, gpt2
 
def train( model, train_dataloader, val_dataloader, optimizer, scheduler=None, clip_grad=None, loss_fn=nn.CrossEntropyLoss(), callbacks=None, device=None, evaluation=True, n_epochs=1, n_batch_per_report=5, accuracy_fn=None, lr_history=None ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Models are moved in-place (the object itself is modified
    
    batch = next(iter(train_dataloader))
    y_shape = batch[-1].squeeze().shape
    # y_shape is typically a scalar for scalar label, 1D tensor for sequence, and 2D for image segmentation task where the label is a 2D tensor representing an image
    if len(y_shape)==1:
        n_labels_per_instance = 1 # typical case
    elif len(y_shape)==2: # (batch, seq_len) for sequence data
        n_labels_per_instance = y_shape[1] 
    elif len(y_shape)==3: # (batch, height, width) for image segmentation task
        n_labels_per_instance = y_shape[1] * y_shape[2]
    else:
        print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )
 
    history = { "total_train_loss":[], "average_train_loss":[], "train_accuracy":[], "total_val_loss":[], "average_val_loss":[], "val_accuracy":[] }
    best_val_loss = float("inf")
    best_state = None
 
    print(f'Start training on device {device}.\n')
    for epoch in range(1, n_epochs + 1):
        if isinstance(loss_fn, nn.MSELoss):
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Train MAE':^9} | {'Val MAE':^9} | {'Elapsed':^9}")
        else:
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Train Acc':^9} | {'Val Acc':^9} | {'Elapsed':^9}")        
        print("-" * 80)
        
        model.train()
        t0_epoch, t0_batch = time.time(), time.time()
        total_correct, total_loss, total_abs_err, weighted_accuracy, batch_loss, batch_count = 0, 0, 0, 0, 0, 0
    
        for batch in train_dataloader:
            batch_count += 1
            
            if isinstance(loss_fn, nn.MSELoss): # regression
                y = batch[-1].to(device)
            else: # classification task
                y = batch[-1].long().to(device) # convert to long (default integer type in PyTorch), required by nn.CrossEntropyLoss()
                
            if len(batch)>2: # example: batch is a triple (sequence, attention mask, label)                
                x = tuple(t.to(device) for t in batch[:-1])
            else:                
                x = batch[0].to(device)
                
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(x).squeeze() # squeeze for regression; otherwise, broadcasting amplifies loss and abs_err
            
            if len(y_shape)==1 or len(y_shape)==3: # scalar target or 2D image target
                loss = loss_fn( logits, y )
            elif len(y_shape)==2: # sequence target
                loss = loss_fn( logits.flatten(0, 1), y.flatten() ) # pred: (batch * seq_len, vocab_size), label: (batch * seq_len,)
            else:
                print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )
                
            batch_loss = loss.item() # item() automatically moves a single-element tensor's value to the CPU and returns it as a standard Python number.
            total_loss += batch_loss * y.shape[0] # Default for CrossEntropyLoss is 'mean' which averages over each loss element in the batch
            
            if isinstance(loss_fn, nn.MSELoss):
                abs_err = torch.sum( torch.abs(logits-y) ).item()
                total_abs_err += abs_err
                batch_accuracy = abs_err / y.shape[0]
            elif accuracy_fn is None:
                if len(y_shape)==1 or len(y_shape)==3: # scalar target or 2D image target
                    preds = torch.argmax(logits, dim=1)
                elif len(y_shape)==2:
                    preds = torch.argmax(logits, dim=2) # sequence target
                else:
                    print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )
                batch_accuracy = (preds == y).cpu().numpy().mean() * 100
                total_correct += (preds == y).sum().item() # item() automatically moves a single-element tensor's value to the CPU and returns it as a standard Python number.
            else:
                batch_accuracy = accuracy_fn( logits, y ).item() * 100
                weighted_accuracy += batch_accuracy * len(y)
                        
            loss.backward()
            
            if clip_grad is not None: # gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
 
            optimizer.step() # Update parameters
            if scheduler is not None:
                scheduler.step() # Update learning rate
            if lr_history is not None:
                lr_history.append( optimizer.param_groups[0]['lr'] )
                        
            if batch_count % n_batch_per_report == 0:
                time_elapsed = time.time() - t0_batch
                print( f"{epoch:^7} | {batch_count:^7} | {batch_loss:^12.6f} | {'-':^10} | {batch_accuracy:^9.2f} | {'-':^9} | {time_elapsed:^9.2f}")                           
                
            batch_loss = 0
            t0_batch = time.time()
 
        if n_batch_per_report < batch_count:
            print("-" * 80)
            
        history["total_train_loss"].append( total_loss )
        history['average_train_loss'].append( history["total_train_loss"][-1] / len(train_dataloader.dataset) )
        
        if isinstance(loss_fn, nn.MSELoss):
            history["train_accuracy"].append( total_abs_err / len(train_dataloader.dataset) ) 
        elif accuracy_fn is None: # scalar target (1), sequence prediction (2), image segmentation (3)
            history["train_accuracy"].append( 100 * total_correct / ( len(train_dataloader.dataset)*n_labels_per_instance ) )
        else:
            history["train_accuracy"].append( weighted_accuracy / len(train_dataloader.dataset) ) # batch-size-weighted accuracy
                
        if evaluation == True:            
            model.eval()
            val_correct, val_loss, total_abs_err, weighted_accuracy = 0, 0, 0, 0
            for batch in val_dataloader:
                if isinstance(loss_fn, nn.MSELoss): # regression task
                    y = batch[-1].to(device)
                else: # classification task
                    y = batch[-1].long().to(device) # convert to long (default integer type in PyTorch), required by nn.CrossEntropyLoss()                    
                if len(batch)>2: # example: (sequence, attention mask)
                    x = tuple(t.to(device) for t in batch[:-1])
                else:
                    x = batch[0].to(device)
                    
                with torch.no_grad(): # switch off autograd using the torch.no_grad() context manager
                    logits = model( x ).squeeze() # squeeze for regression; otherwise, broadcasting amplifies loss and abs_err
                    
                if len(y_shape)==1 or len(y_shape)==3: # scalar target or 2D image target
                    loss = loss_fn( logits, y )
                elif len(y_shape)==2: # sequence target
                    loss = loss_fn( logits.flatten(0, 1), y.flatten() ) # pred: (batch * seq_len, vocab_size), label: (batch * seq_len,)
                else:
                    print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )
                    
                val_loss += loss.item() * len(y)  # Default for CrossEntropyLoss is 'mean' which averages over each loss element in the batch
                
                if isinstance(loss_fn, nn.MSELoss):
                    total_abs_err += torch.sum( torch.abs(logits-y) ).item()
                elif accuracy_fn is None:
                    if len(y_shape)==1 or len(y_shape)==3: # scalar target or 2D image target
                        preds = torch.argmax(logits, dim=1)
                    elif len(y_shape)==2:
                        preds = torch.argmax(logits, dim=2) # sequence target
                    else:
                        print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )                    
                    val_correct += (preds == y).cpu().numpy().sum().item()
                else:
                    batch_accuracy = accuracy_fn(logits, y).item()
                    weighted_accuracy += batch_accuracy * len(y)
                
            history["total_val_loss"].append( val_loss )
            history['average_val_loss'].append( history["total_val_loss"][-1] / len(val_dataloader.dataset) )
            
            if isinstance(loss_fn, nn.MSELoss):
                history["val_accuracy"].append( total_abs_err / len(val_dataloader.dataset) ) 
            elif accuracy_fn is None: # scalar target (1), sequence prediction (2), image segmentation (3)
                history["val_accuracy"].append( 100 * val_correct / (len(val_dataloader.dataset)*n_labels_per_instance) )
            else:
                history["val_accuracy"].append( 100 * weighted_accuracy / len(val_dataloader.dataset) ) # batch-size-weighted accuracy
            
            time_elapsed = time.time() - t0_epoch    
            print( f"{epoch:^7} | {'-':^7} | {history['average_train_loss'][-1]:^12.6f} | {history['average_val_loss'][-1]:^10.6f} | {history['train_accuracy'][-1]:^9.2f} | {history['val_accuracy'][-1]:^9.2f} | {time_elapsed:^9.2f}")
            # ModelCheckpoint(save_best_only=True, monitor="val_loss")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy( model.state_dict() )
                optimizer_state = copy.deepcopy( optimizer.state_dict() )
                if callbacks:
                    for cb in callbacks:
                        cb.on_best(model, optimizer)
        else:
            print( f"{epoch:^7} | {'-':^7} | {history['total_train_loss'][-1]:^12.6f} | {'-':^10} | {history['train_accuracy'][-1]:^9.2f} | {'-':^10} | {time_elapsed:^9.2f}")
        print("-" * 80)
    # restore best, similar to Keras's best-checkpoint behavior
    if best_state is not None:
        model.load_state_dict(best_state)
        optimizer.load_state_dict(optimizer_state)
    return history
 
 
def evaluate(model, loader, device, loss_fn=nn.CrossEntropyLoss(), accuracy_fn=None):
    model.eval()
    batch = next(iter(loader))
    y_shape = batch[-1].squeeze().shape
    if len(y_shape)==1:
        n_labels_per_instance = 1 # typical case
    elif len(y_shape)==2: # (batch, seq_len) for sequence data
        n_labels_per_instance = y_shape[1] 
    elif len(y_shape)==3: # (batch, height, width) for image segmentation task
        n_labels_per_instance = y_shape[1] * y_shape[2]
    else:
        print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )
        
    val_correct, val_loss, total_abs_err, weighted_accuracy = 0,0,0,0
    for batch in loader:
        if isinstance(loss_fn, nn.MSELoss): # regression task
            y = batch[-1].to(device)
        else: # classification task
            y = batch[-1].long().to(device) # convert to long (default integer type in PyTorch), required by nn.CrossEntropyLoss()
        if len(batch)>2: # example: (sequence, attention mask)
            x = tuple(t.to(device) for t in batch[:-1])
        else:
            x = batch[0].to(device)
            
        with torch.no_grad(): # switch off autograd using the torch.no_grad() context manager
            logits = model(x).squeeze() # squeeze for regression; otherwise, broadcasting amplifies loss and abs_err
            
        if len(y_shape)==1 or len(y_shape)==3: # scalar target or 2D image target
            loss = loss_fn( logits, y )
        elif len(y_shape)==2: # sequence target
            loss = loss_fn( logits.flatten(0, 1), y.flatten() ) # pred: (batch * seq_len, vocab_size), label: (batch * seq_len,)
        else:
            print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )
            
        val_loss += loss.item() * y.shape[0]
        
        if isinstance(loss_fn, nn.MSELoss):
            total_abs_err += torch.sum( torch.abs(logits-y) ).item()
        elif accuracy_fn is None:
            if len(y_shape)==1 or len(y_shape)==3: # scalar target or 2D image target
                preds = torch.argmax(logits, dim=1)
            elif len(y_shape)==2:
                preds = torch.argmax(logits, dim=2) # sequence target: (batch, seq_len, vocab_size)
            else:
                print( f"Error: Targe tensors with {len(y_shape)} or more axes are currently not supported!" )              
            val_correct += (preds == y).cpu().numpy().sum()
        else:
            batch_accuracy = accuracy_fn(logits, y).item()
            weighted_accuracy += batch_accuracy * len(y)
            
    val_loss = val_loss / len(loader.dataset)
    
    if isinstance(loss_fn, nn.MSELoss):
        val_accuracy = total_abs_err / len(loader.dataset)
    elif accuracy_fn is None:
        val_accuracy = val_correct/ (len(loader.dataset)**n_labels_per_instance)
    else:
        val_accuracy = 100 * weighted_accuracy / len(loader.dataset) # batch-size-weighted accuracy
        
    return val_loss, val_accuracy
 