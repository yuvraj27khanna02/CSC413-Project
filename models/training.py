import pandas as pd
import torch
from arch import *
from data_preprocessing import load_data
import json
from datetime import datetime

def json_write(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def json_update(data, filepath):
    with open(filepath, 'r') as f:
        old_data = json.load(f)
    old_data.extend(data)
    with open(filepath, 'w') as f:
        json.dump(old_data, f)

def get_device() -> torch.device:
    """ Returns the appropriate device based on MPS, CUDA, or CPU in order    

    original:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

    cuda/cpu only:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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


def get_laptime_accuracy(dataset:torch.utils.data.TensorDataset, model:torch.nn.Module,
                         output_mae_accuracy=True, output_mse_accuracy=False, output_rmse_accuracy=False):
    total_mae_accuracy = 0
    total_mse_accuracy = 0
    total_rmse_accuracy = 0
    total_count = 0

    for (X, t_laptime, t_position) in dataset:
        t_laptime = t_laptime.view(-1, 1)
        y_laptime, y_position = model(X)

        mae_accuracy = torch.mean(torch.abs(y_laptime - t_laptime))
        mse_accuracy = torch.mean((y_laptime - t_laptime)**2)
        rmse_accuracy = torch.sqrt(mse_accuracy)

        total_mae_accuracy += mae_accuracy
        total_mse_accuracy += mse_accuracy
        total_rmse_accuracy += rmse_accuracy
        total_count += 1
    
    if output_mae_accuracy and output_mse_accuracy and output_rmse_accuracy:
        return total_mae_accuracy / total_count, total_mse_accuracy / total_count, total_rmse_accuracy / total_count
    elif output_mae_accuracy:
        return total_mae_accuracy / total_count
    else:
        raise ValueError("No output specified")

def get_position_accuracy(dataset:torch.utils.data.TensorDataset, model:torch.nn.Module):
    total_correct = 0
    total_count = 0
    for (X, t_laptime, t_position) in dataset:
        y_laptime, y_position = model(X)
        total_correct += int(torch.sum(y_position == t_position))
        total_count += t_position.shape[0]
    return total_correct / total_count

def get_laptime_accuracy_v2(dataset, laptime_model):
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    total_count = 0
    for (X, t_laptime, _) in dataset:
        t_laptime = t_laptime.view(-1, 1)
        y_laptime = laptime_model(X)
        mae_accuracy = torch.mean(torch.abs(y_laptime - t_laptime))
        mse_accuracy = torch.mean((y_laptime - t_laptime)**2)
        rmse_accuracy = torch.sqrt(mse_accuracy)
        total_mae += mae_accuracy
        total_mse += mse_accuracy
        total_rmse += rmse_accuracy
        total_count += 1
    return total_mae / total_count, total_mse / total_count, total_rmse / total_count

def get_position_accuracy_v2(dataset, position_model):
    total_correct = 0
    total_count = 0
    for (X, _, t_position) in dataset:
        y_position = position_model(X)
        total_correct += int(torch.sum(y_position == t_position))
        total_count += t_position.shape[0]
    return total_correct / total_count

def get_data(n=3, train_split=0.6, val_split=0.2, data_dim=int,
             device=get_device(), small_data=False, VERBOSE=False):
    
    filepath = f"ready_data/ngrams_data_{n}_{data_dim}.pth"
    tensor_data = load_data(file_path=filepath)
    X_tensor, t_tensor = tensor_data.tensors
    X_tensor, t_tensor = X_tensor.to(device), t_tensor.to(device)

    if small_data:
        X_tensor = X_tensor[:10000]
        t_tensor = t_tensor[:10000]

    t_laptime = t_tensor[:, 1]
    t_postion = t_tensor[:, 57:77]
    total_count = X_tensor.shape[0]
    train_count, val_count = int(total_count * train_split), int(total_count * (train_split+val_split))
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor[:train_count], t_laptime[:train_count], t_postion[:train_count])
    val_dataset = torch.utils.data.TensorDataset(X_tensor[train_count:val_count], t_laptime[train_count:val_count], t_postion[train_count:val_count])
    test_dataset = torch.utils.data.TensorDataset(X_tensor[val_count:], t_laptime[val_count:], t_postion[val_count:])
    if VERBOSE:
        print(f"train size: {len(train_dataset)} \t val size: {len(val_dataset)} \t test size: {len(test_dataset)}")
        print(f"train device:{train_dataset.tensors[0].device} \t val device:{val_dataset.tensors[0].device} \t test device:{test_dataset.tensors[0].device}")
    
    return train_dataset, val_dataset, test_dataset

def train_model(model:ANN_MIMO_v2,
                train_dataloader:torch.utils.data.DataLoader,
                val_dataloader:torch.utils.data.DataLoader,
                num_epochs=10, 
                verbose_every=10,
                criterion_laptime=torch.nn.MSELoss(),
                criterion_position=torch.nn.CrossEntropyLoss(),
                device=get_device(),
                datatype=torch.float32,
                loss_model=Loss_Model_v1,
                VERBOSE=False):
    
    if VERBOSE:
        print(f"Training on {device}")
    
    model.to(device, dtype=datatype)
    loss_model.to(device, dtype=datatype)

    train_loss_sum_list = []
    train_loss_laptime_list = []
    train_loss_position_list = []
    train_acc_laptime_list = []
    train_acc_position_list = []
    val_acc_laptime_list = []
    val_acc_position_list = []
    epoch_list = []
    itr_list = []
    model_parameters_list = []
    loss_model_parameters_list = []
    
    model_optimiser = model.get_optimiser()
    loss_optimiser = loss_model.get_optimiser()
    iter_count = 0

    for epoch in range(num_epochs):
        for (X, t_laptime, t_position) in train_dataloader:

            # Format data
            X = X.to(device, dtype=datatype)
            t_laptime = t_laptime.to(device, dtype=datatype)
            t_position = t_position.to(device, dtype=datatype)
            t_laptime = t_laptime.view(-1, 1)

            # Training model
            model.train()
            # Forward pass
            y_laptime, y_position = model(X)
            loss_laptime = criterion_laptime(y_laptime, t_laptime)
            loss_position = criterion_position(y_position, t_position)
            loss = loss_model(loss_laptime, loss_position)
            # Backward pass
            loss.backward()
            loss_optimiser.step()
            loss_optimiser.zero_grad()
            model_optimiser.step()
            model_optimiser.zero_grad()

            # Evaluating model
            with torch.no_grad():
                model.eval()
                train_laptime_accuracy = get_laptime_accuracy(train_dataloader, model)
                train_position_accuracy = get_position_accuracy(train_dataloader, model)
                val_laptime_accuracy = get_laptime_accuracy(val_dataloader, model)
                val_position_accuracy = get_position_accuracy(val_dataloader, model)
            
            train_loss_sum_list.append(loss.item())
            train_loss_laptime_list.append(loss_laptime.item())
            train_loss_position_list.append(loss_position.item())
            train_acc_laptime_list.append(train_laptime_accuracy)
            train_acc_position_list.append(train_position_accuracy)
            val_acc_laptime_list.append(val_laptime_accuracy)
            val_acc_position_list.append(val_position_accuracy)
            epoch_list.append(epoch)
            itr_list.append(iter_count)
            # TODO: add model params to model_parameters_list
            loss_model_parameters_list.append(dict(loss_model.state_dict()))

            if (VERBOSE) and (iter_count % verbose_every == 0):
                print(f"epoch:{epoch} \t itr:{iter_count} \t batch size:{y_laptime.shape[0]} \t loss:{loss.item()}"
                      f"\n LOSS MODEL \t w1:{loss_model.w1.item()} \t w2:{loss_model.w2.item()}"
                      f"\n LAPTIME OUTPUT \t train mse loss: {loss_laptime.item()} \t train mae accuracy: {train_laptime_accuracy} \t val mae accuracy: {val_laptime_accuracy}"
                      f"\n POSITION OUTPUT \t train loss: {loss_position.item()} \t train accuracy: {train_position_accuracy} \t val accuracy: {val_position_accuracy} \n")

            iter_count += 1
    
    return {
        'laptime_loss_criterion': criterion_laptime,
        'position_loss_criterion': criterion_position,
        'train_loss_sum': train_loss_sum_list,
        'train_loss_laptime': train_loss_laptime_list,
        'train_loss_position': train_loss_position_list,
        'train_acc_laptime': train_acc_laptime_list,
        'train_acc_position': train_acc_position_list,
        'val_acc_laptime': val_acc_laptime_list,
        'val_acc_position': val_acc_position_list,
        'epoch': epoch_list,
        'itr': itr_list,
    }

def train_model_v2(laptime_model: MLP_regression_v1,
                   position_model: MLP_MC_v1,
                   train_dataset,
                   val_dataset,
                   num_epochs=10,
                   verbose_every=10,
                   criterion_laptime=torch.nn.MSELoss(),
                   criterion_position=torch.nn.CrossEntropyLoss(),
                   laptime_optimiser=str,
                   position_optimiser=str,
                   device=get_device(),
                   datatype=torch.float32,
                   weight_decay=0.001,
                   lr=1e-3,
                   batch_size=32,
                   VERBOSE=False):
    
    if VERBOSE:
        print(f"Training on {device}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    if VERBOSE:
        X, t_lt, t_p = next(iter(train_dataloader))
        print(f"X={X.shape} \t t_laptime={t_lt.shape} \t t_position={t_p.shape}")
        print(f"X device:{X.device} \t t_laptime device:{t_lt.device} \t t_position device:{t_p.device}")
    
    laptime_model.to(device)
    position_model.to(device)

    train_loss_laptime_list = []
    train_loss_position_list = []
    train_acc_laptime_list = []
    val_acc_laptime_list = []
    train_acc_position_list = []
    val_acc_position_list = []
    epoch_list = []
    itr_list = []
    laptime_model_params_list = []
    position_model_params_list = []
    epoch_time_delta_list = []
    iter_time_delta_list = []

    laptime_optimiser = _get_optimiser(optimiser=laptime_optimiser, model_params=laptime_model.parameters(), lr=lr, weight_decay=weight_decay) 
    position_optimiser = _get_optimiser(optimiser=position_optimiser, model_params=position_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    iter_count = 0
    train_start_time = datetime.now()
    prev_epoch_time = train_start_time

    for epoch in range(num_epochs):

        prev_itr_time = datetime.now()

        for (X, t_laptime, t_position) in train_dataloader:

            # Format data
            X = X.to(device, dtype=datatype)
            t_laptime = t_laptime.to(device, dtype=datatype)
            t_position = t_position.to(device, dtype=datatype)
            t_laptime = t_laptime.view(-1, 1)
            # Training model
            laptime_model.train()
            position_model.train()
            # Forward pass
            y_laptime = laptime_model(X)
            y_position = position_model(X)
            loss_laptime = criterion_laptime(y_laptime, t_laptime)
            loss_position = criterion_position(y_position, t_position)
            # Backward pass
            loss_laptime.backward()
            laptime_optimiser.step()
            laptime_optimiser.zero_grad()
            loss_position.backward()
            position_optimiser.step()
            position_optimiser.zero_grad()

            # Evaluating model
            with torch.no_grad():
                laptime_model.eval()
                position_model.eval()

                train_laptime_mae, train_laptime_mse, train_laptime_rmse = get_laptime_accuracy_v2(train_dataloader, laptime_model)
                val_laptime_mae, val_laptime_mse, val_laptime_rmse = get_laptime_accuracy_v2(val_dataloader, laptime_model)
                train_position_accuracy = get_position_accuracy_v2(train_dataloader, position_model)
                val_position_accuracy = get_position_accuracy_v2(train_dataloader, position_model)

            train_loss_laptime_list.append(loss_laptime.item())
            train_loss_position_list.append(loss_position.item())

            train_acc_position_list.append((train_laptime_mae.item(), train_laptime_mse.item(), train_laptime_rmse.item()))
            val_acc_position_list.append((val_laptime_mae.item(), val_laptime_mse.item(), val_laptime_rmse.item()))
            
            train_acc_position_list.append(train_position_accuracy)
            val_acc_position_list.append(val_position_accuracy)
            epoch_list.append(epoch)
            itr_list.append(iter_count)
            laptime_model_params_list.append(laptime_model.parameters())
            position_model_params_list.append(position_model.parameters())

            
            if (VERBOSE) and (iter_count % verbose_every == 0):
                print(f"epoch:{epoch} \t itr:{iter_count} \t batch size:{y_laptime.shape[0]}"
                      f"\n LAPTIME OUTPUT \t train MSE Loss: {loss_laptime.item()} \t train accuracy: {train_laptime_mae} , {train_laptime_mse} , {train_laptime_rmse} \t val accuracy: {val_laptime_mae} , {val_laptime_mse} , {val_laptime_rmse} \t"
                      f"\n POSITION OUTPUT \t train CE Loss: {loss_position.item()} \t train accuracy: {train_position_accuracy} \t val accuracy: {val_position_accuracy} \n")
                
            iter_count += 1

            curr_itr_time = datetime.now()
            this_itr_time = curr_itr_time - prev_itr_time
            iter_time_delta_list.append({
                'seconds': this_itr_time.seconds,
                'microseconds': this_itr_time.microseconds
            })
            prev_itr_time = curr_itr_time
        
        curr_epoch_time = datetime.now()
        this_epoch_time = curr_epoch_time - prev_epoch_time
        epoch_time_delta_list.append({
            'seconds': this_epoch_time.seconds,
            'microseconds': this_epoch_time.microseconds
        })
        prev_epoch_time = curr_epoch_time
    
    total_time = datetime.now() - train_start_time

    return {
        'device': str(device),
        'total_time': {
            'seconds': total_time.seconds,
            'microseconds': total_time.microseconds
        },
        'model_info': {
            'laptime_model': str(laptime_model),
            'position_model': str(position_model),
            'laptime_model_summary': str(get_model_summary(laptime_model)),
            'position_model_summary': str(get_model_summary(position_model))            
        },
        'data_info': {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'input_shape': X.size(),
            'laptime_shape': t_laptime.size(),
            'position_shape': t_position.size(),
        },
        'hyperparameters': {
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': num_epochs,
            'weight_decay': weight_decay,
            'dropout': None,
            'laptime_optimiser': str(laptime_optimiser),
            'position_optimiser': str(position_optimiser)
        },
        'metrics': {
            'epoch_list': epoch_list,
            'itr_list': itr_list,
            'train_loss_laptime': train_loss_laptime_list,
            'train_loss_position': train_loss_position_list,
            'train_acc_laptime': train_acc_laptime_list,
            'train_acc_position': train_acc_position_list,
            'val_acc_laptime': val_acc_laptime_list,
            'val_acc_position': val_acc_position_list,
            'epoch_time': epoch_time_delta_list,
            'iter_time': iter_time_delta_list
        }
    }

if __name__ == "__main__":

    train_begin_time = datetime.now()

    # model1 = ANN_MIMO_v2(input_num=2, input_size=125, hidden_dim=100, emb_dim=30, hidden_output_list=[5, 25], act_fn='relu', optimiser='adam', lr=0.0001)
    # loss_model1 = Loss_Model_v1(optimiser='adam', lr=0.1)
    # print(f"\n === Training model === \n")
    # metrics = train_model(model=model1, loss_model=loss_model1, train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=10, verbose_every=100, VERBOSE=True)

    all_metrics = []

    # test to compare training on cpu and mps

    device = torch.device('cpu')
    data_dim = 124

    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:

        train_dataset, val_dataset, test_dataset = get_data(n=3, small_data=True, data_dim=124, VERBOSE=True, device=device)

        hidden_size_list = [(10, 30), (20, 50), (50, 100), (100, 150), (150, 200), (200, 250)]

        for laptime_model_hidden_size, position_model_hidden_size in hidden_size_list:
            laptime_model = MLP_regression_v1(input_size=250, hidden_size=laptime_model_hidden_size, act_fn='relu', data_dim=data_dim)
            position_model = MLP_MC_v1(input_size=250, hidden_size=50, output_classes=20, act_fn='relu', data_dim=data_dim)
            print("\n === Training model 1 === \n")
            metrics = train_model_v2(laptime_model=laptime_model, position_model=position_model,
                                    train_dataset=train_dataset, val_dataset=val_dataset,
                                    laptime_optimiser='adam', position_optimiser='adam',
                                    verbose_every=100, num_epochs=10, batch_size=32, lr=1e-5, weight_decay=0.001,
                                    VERBOSE=True, device=device)
            all_metrics.append(metrics)
        
        for num_layers in [2, 3, 5, 10, 15, 20]:
            laptime_model = ANN_regression_v1(input_size=250, num_layers=num_layers, hidden_size=50, act_fn='relu')
            position_model = ANN_MC_v1(input_size=250, hidden_size=50, num_layers=num_layers, output_classes=20, act_fn='relu')
            print('\n === Training model 2 === \n')
            metrics_2 = train_model_v2(laptime_model=laptime_model, position_model=position_model,
                                    train_dataset=train_dataset, val_dataset=val_dataset,
                                    laptime_optimiser='adam', position_optimiser='adam',
                                    verbose_every=100, num_epochs=10, batch_size=32, lr=1e-5, weight_decay=0.001,
                                    VERBOSE=True, device=device)
            all_metrics.append(metrics_2)

    # to create new metrics file
    json_write(all_metrics, 'obs/train_metrics.json')

    print(f"\n \t === Training complete === \n"
          f"total time: {datetime.now() - train_begin_time} \t metrics count: {len(all_metrics)} \n")

    # to write over existing metrics file
    # json_update(all_metrics, 'obs/train_metrics.json')


