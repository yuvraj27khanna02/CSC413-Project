import pandas as pd
import torch
from arch import *
from RNN import *
from data_preprocessing import load_data
import json
from datetime import datetime
from race_lap_ngrams import RaceLapNgrams

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

def laptime_accuracy_rnn(model, dataset, device):
    mae, mse, rmse, count = 0, 0, 0, 0
    total_samples = len(dataset)

    for X, t_laptime, _ in dataset:
        X = X.to(device)
        t_laptime = t_laptime.to(device)

        y_laptime, _ = model(X)

        mae += torch.mean(torch.abs(y_laptime - t_laptime)).sum().item()
        mse += torch.mean((y_laptime - t_laptime) ** 2).sum()
        rmse += torch.sqrt(mse)
        count += 1
    
    return mae / count, mse.item() / count, rmse.item() / count

def position_accuracy_rnn(model, dataset, device):
    correct, total = 0, 0
    for X, _, t_position in dataset:
        X = X.to(device)
        t_position = t_position.to(device).view(-1)
        y_position, _ = model(X)

        _, y_classes = torch.max(y_position, 1)
        _, t_classes = torch.max(t_position, 0)

        correct += (y_classes == t_classes).sum().item()
        total += t_position.size(0)

    return correct / total

def train_model_rnn(laptime_model: RNN_regression_v1,
                   position_model: RNN_MC_v1,
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
        print(f"X={X.shape} \t t_laptime={t_lt.shape} \t t_position={t_p.shape} \t device={device}")
    
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

    laptime_hidden = None
    position_hidden = None

    for epoch in range(num_epochs):

        prev_itr_time = datetime.now()

        for (X, t_laptime, t_position) in train_dataloader:

            # Format data
            X = X.to(device, dtype=datatype)
            t_laptime = t_laptime.to(device, dtype=datatype)
            t_position = t_position.to(device, dtype=datatype)
            t_laptime = t_laptime.view(-1, 1)
            X = X.view(batch_size, -1, X.size(-1))
            _, t_position_indices = torch.max(t_position, dim=1)
            # Training model
            laptime_model.train()
            position_model.train()
            # Forward pass
            y_laptime, laptime_hidden = laptime_model(X)
            y_position, position_hidden = position_model(X)

            laptime_hidden = laptime_hidden.detach()            # what does this do?
            position_hidden = position_hidden.data              # what does this do?

            loss_laptime = criterion_laptime(y_laptime, t_laptime)
            loss_position = criterion_position(y_position, t_position_indices)
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
                train_laptime_mae, train_laptime_mse, train_laptime_rmse = laptime_accuracy_rnn(laptime_model, train_dataset, device)
                val_laptime_mae, val_laptime_mse, val_laptime_rmse = laptime_accuracy_rnn(laptime_model, val_dataset, device)
                train_position_accuracy = position_accuracy_rnn(position_model, train_dataset, device)
                val_position_accuracy = position_accuracy_rnn(position_model, val_dataset, device)

            train_acc_laptime_list.append({
                'mae': train_laptime_mae,
                'mse': train_laptime_mse,
                'rmse': train_laptime_rmse
            })
            val_acc_laptime_list.append({
                'mae': val_laptime_mae,
                'mse': val_laptime_mse,
                'rmse': val_laptime_rmse
            })
            train_loss_laptime_list.append(loss_laptime.item())
            train_loss_position_list.append(loss_position.item())
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
            'iter_time': iter_time_delta_list,
            # 'laptime_hidden': laptime_hidden,
            # 'position_hidden': position_hidden
        }
    }

if __name__ == "__main__":

    train_begin_time = datetime.now()

    all_metrics = []

    n = 3
    input_num = n-1

    race_lap_ngrams = RaceLapNgrams(n=n)
    race_lap_ngrams.split_by_proportion()

    train_dataset = torch.utils.data.TensorDataset(*race_lap_ngrams.get_train_tensors())
    val_dataset = torch.utils.data.TensorDataset(*race_lap_ngrams.get_val_tensors())
    test_dataset = torch.utils.data.TensorDataset(*race_lap_ngrams.get_test_tensors())
    
    device = torch.device('cpu')
    data_dim = race_lap_ngrams.data_dim

    hidden_size_list = [(15, 15), (25, 25)]
    num_layers_list = [2, 10]
    lr = 1e-5

    for laptime_model_hidden_size, position_model_hidden_size in hidden_size_list:
        print(f'\n \t === training hidden size: {laptime_model_hidden_size} , {position_model_hidden_size} ===')
        for num_layer in num_layers_list:
            print(f'\n \t === lr: {lr} \t num_layers: {num_layer} \t training hidden size: {laptime_model_hidden_size} , {position_model_hidden_size} ======')

            laptime_model = RNN_regression_v1(input_size=data_dim, input_num=input_num, emb_size=laptime_model_hidden_size, num_layers=num_layer, hidden_size=laptime_model_hidden_size, act_fn='relu')
            position_model = RNN_MC_v1(input_size=data_dim, input_num=input_num, emb_size=position_model_hidden_size, hidden_size=position_model_hidden_size, num_layers=num_layer, output_classes=20, act_fn='relu')
            rnn_v2_metrics = train_model_rnn(laptime_model=laptime_model, position_model=position_model,
                                    train_dataset=train_dataset, val_dataset=val_dataset,
                                    laptime_optimiser='adam', position_optimiser='adam',
                                    verbose_every=30, num_epochs=1, batch_size=64, lr=lr, weight_decay=0,
                                    VERBOSE=True, device=device)
            all_metrics.append(rnn_v2_metrics)

    # to create new metrics file
    json_write(all_metrics, 'train_metrics.json')

    print(f"\n \t === Training complete === \n"
          f"total time: {datetime.now() - train_begin_time} \t metrics count: {len(all_metrics)} \n")

    # to write over existing metrics file
    # json_update(all_metrics, 'obs/train_metrics.json')



