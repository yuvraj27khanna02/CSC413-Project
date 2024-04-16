import pandas as pd
import torch
from models.arch import *
import json
from datetime import datetime
from models.race_lap_ngrams import RaceLapNgrams
import matplotlib.pyplot as plt

def _get_optimiser(optimiser=str, model_params=dict, lr=float, weight_decay=float):
    """ optim list: ['adam', 'sgd', 'adamw', 'rmsprop']
    """
    if optimiser == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    if optimiser == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    if optimiser == 'adamw':
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    if optimiser == 'rmsprop':
        return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)

def get_accuracy(laptime_model, position_model, dataset, device=torch.device, max_val=1000):
    laptime_mae, laptime_mse, laptime_rmse, position_correct, total = 0, 0, 0, 0, 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for i,(X, t_laptime, t_position) in enumerate(dataloader):
        X = X.to(device)
        t_laptime = t_laptime.to(device)
        t_position = t_position.to(device)
        t_laptime = t_laptime.view(-1, 1)
        y_laptime = laptime_model(X)
        laptime_mae += float(torch.sum(torch.abs(y_laptime - t_laptime)))
        laptime_mse += float(torch.sum((y_laptime - t_laptime)**2))
        laptime_rmse += float(torch.sum(torch.sqrt((y_laptime - t_laptime)**2)))
        z_position = position_model(X)
        y_position = torch.argmax(z_position, axis=1)
        t_position = torch.argmax(t_position, axis=1)
        position_correct += int(torch.sum(y_position == t_position))
        total += X.size(0)
        if i >= max:
            break
    return laptime_mae/total, laptime_mse/total, laptime_rmse/total, position_correct/total
    
def train_model(laptime_model:torch.nn.Module, position_model:torch.nn.Module, train_data, val_data, num_epochs=int,
                     learning_rate=float, batch_size=int, weight_decay=0, 
                     device=torch.device, datatype=torch.float32,
                     laptime_criterion=torch.nn.MSELoss(), position_criterion=torch.nn.CrossEntropyLoss(),
                     laptime_optimiser=str, position_optimiser=str,
                     save_every=int, verbose_every=int, VERBOSE=False):
    
    laptime_loss_list, position_loss_list = [], []
    laptime_train_mae_list, laptime_train_mse_list, laptime_train_rmse_list, position_train_acc_list = [], [], [], []
    laptime_val_mae_list, laptime_val_mse_list, laptime_val_rmse_list, position_val_acc_list = [], [], [], []
    iter_count_list, epoch_count_list, iter_time_list, epoch_time_list = [], [], [], []

    laptime_model = laptime_model.to(device=device, dtype=datatype)
    position_model = position_model.to(device=device, dtype=datatype)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    laptime_optimiser = _get_optimiser(laptime_optimiser, laptime_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    position_optimiser = _get_optimiser(position_optimiser, position_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if VERBOSE:
        X, t_laptime, t_position = train_data[0]
        print(f"train size:{len(train_data)}\tval size:{len(val_data)}\tX shape:{X.shape}\tt_laptime shape:{t_laptime.shape}\tt_position shape:{t_position.shape}\n"
              f"epochs:{num_epochs}\tbatch_size:{batch_size}\tverbose_every:{verbose_every}\tsave_every:{save_every}\tdevice:{device}\n"
              f"laptime loss criterion:{laptime_criterion}\tposition loss criterion:{position_criterion}\n"
              f"laptime optimiser:{laptime_optimiser}\tposition optimiser:{position_optimiser}\n"
              f"Laptime model:{laptime_model}\nPosition model:{position_model}\n")
    
    iter_count = 0
    train_start_time = datetime.now()
    prev_epoch_time = train_start_time
    prev_iter_time = train_start_time

    for e in range(num_epochs):

        for i, (X, t_laptime, t_position) in enumerate(train_dataloader):
            X = X.to(device)
            t_laptime = t_laptime.to(device)
            t_position = t_position.to(device)
            t_laptime = t_laptime.view(-1, 1)

            laptime_model.train()
            position_model.train()

            z_laptime = laptime_model(X)
            z_position = position_model(X)

            laptime_loss = laptime_criterion(z_laptime, t_laptime)
            laptime_loss.backward()
            laptime_optimiser.step()
            laptime_optimiser.zero_grad()

            position_loss = position_criterion(z_position, t_position)
            position_loss.backward()
            position_optimiser.step()
            position_optimiser.zero_grad()

            if VERBOSE and (iter_count % verbose_every == 0 or iter_count % save_every == 0):

                laptime_loss_list.append(float(laptime_loss.item()))
                position_loss_list.append(float(position_loss.item()))

                with torch.no_grad():
                    laptime_model.eval()
                    position_model.eval()
                    laptime_train_mae, laptime_train_mse, laptime_train_rmse, position_train_acc = get_accuracy(laptime_model, position_model, train_data, device)
                    laptime_val_mae, laptime_val_mse, laptime_val_rmse, position_val_acc = get_accuracy(laptime_model, position_model, val_data, device)

                laptime_train_mae_list.append(laptime_train_mae)
                laptime_train_mse_list.append(laptime_train_mse)
                laptime_train_rmse_list.append(laptime_train_rmse)
                position_train_acc_list.append(position_train_acc)
                laptime_val_mae_list.append(laptime_val_mae)
                laptime_val_mse_list.append(laptime_val_mse)
                laptime_val_rmse_list.append(laptime_val_rmse)
                position_val_acc_list.append(position_val_acc)
                iter_count_list.append(iter_count)
                epoch_count_list.append(e)

                curr_iter_time = datetime.now()
                this_iter_time = curr_iter_time - prev_iter_time
                iter_time_list.append({
                    'seconds': this_iter_time.seconds,
                    'microseconds': this_iter_time.microseconds
                })
                prev_iter_time = curr_iter_time
    
        curr_epoch_time = datetime.now()
        this_epoch_time = curr_epoch_time - prev_epoch_time
        epoch_time_list.append({
            'seconds': this_epoch_time.seconds,
            'microseconds': this_epoch_time.microseconds
        })
        prev_epoch_time = curr_epoch_time
    
    total_time = datetime.now() - train_start_time

    return {
        'device': str(device),
        'total_time':{
            'seconds': total_time.seconds,
            'microseconds': total_time.microseconds
        },
        'model_info':{
            'laptime_model': str(laptime_model),
            'position_model': str(position_model)
        },
        'data_info': {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'X_shape': X.shape,
            't_laptime_shape': t_laptime.shape,
            't_position_shape': t_position.shape
        },
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': num_epochs,
            'weight_decay': weight_decay,
            'laptime_criterion': str(laptime_criterion),
            'position_criterion': str(position_criterion),
            'laptime_optimiser': str(laptime_optimiser),
            'position_optimiser': str(position_optimiser)
        },
        'metrics':{
            'laptime_loss': laptime_loss_list,
            'position_loss': position_loss_list,

            'laptime_train_mae': laptime_train_mae_list,
            'laptime_train_mse': laptime_train_mse_list,
            'laptime_train_rmse': laptime_train_rmse_list,

            'position_train_acc': position_train_acc_list,

            'laptime_val_mae': laptime_val_mae_list,
            'laptime_val_mse': laptime_val_mse_list,
            'laptime_val_rmse': laptime_val_rmse_list,

            'position_val_acc': position_val_acc_list,

            'iter_count': iter_count_list,
            'epoch_count': epoch_count_list,

            'iter_time': iter_time_list,
            'epoch_time': epoch_time_list
        }
    }

class ModelTrain:

    hyperparam_list = [{
        'n': int,
        'num_epochs': int,
        'batchsize': int,
        'hidden_size': int,
        'emb_size': int,
        'num_layers': int,
        'laptime_model': torch.nn.Module,
        'position_model': torch.nn.Module,
        'learning_rate': float,
        'weight_decay': float,
        'device': torch.device,
        'datatype': torch.dtype,
        'laptime_criterion': torch.nn,
        'position_criterion': torch.nn,
        'laptime_optimiser': str,
        'position_optimiser': str,
        'verbose_every': int,
    }]


    def __init__(self, hyperparam_list:dict):
        self.hyperparam_list = hyperparam_list
        pass

    def add_hyperparam(self, hyperparam:dict):
        self.hyperparam_list.append(hyperparam)
        pass

    def train_on_hyperparam(self):
        print(f"Training on {len(self.hyperparam_list)} hyperparameters")
        for hyperparam in self.hyperparam_list:
            pass
