import pandas as pd
import torch
from arch import ANN_MIMO_v2, Loss_Model_v1
from data_preprocessing import load_data

def get_device() -> torch.device:
    """ Returns the appropriate device based on MPS, CUDA, or CPU in order    

    original:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def generate_ngrams(df, n, columns_to_drop=['Driver_Orig', 'Event_Orig']):
    """generate_ngrams creates n-grams from the input dataframe of consecutive laps
    from a driver at a single event.

    columns_to_drop: list of columns to drop from the n-gram (not tensor friendly)

    Returns a Pandas dataframe of the n-grams
    """
    ngrams_list = []

    grouped = df.groupby(['Year', 'Event_Orig', 'Driver_Orig'])
    for _, group in grouped:
        if columns_to_drop:
            group = group.drop(columns=columns_to_drop, errors='ignore')

        sorted_group = group.sort_values(by='LapNumber')

        for i in range(len(sorted_group) - n + 1):
            potential_ngram = sorted_group.iloc[i:i + n]

            lap_numbers = potential_ngram['LapNumber'].to_list()
            # Check if the n-gram laps are consecutive
            if lap_numbers == list(range(int(min(lap_numbers)), int(min(lap_numbers)) + n)):
                ngrams_list.append(potential_ngram.values.flatten())

    col_names = [f'{col}_{i+1}' for i in range(n) for col in group]
    ngrams_df = pd.DataFrame(ngrams_list, columns=col_names)

    return ngrams_df.dropna()

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

def get_data(n=3, train_split=0.6, val_split=0.2,
             batch_size=32, device=get_device(),
             VERBOSE=False):
    
    filepath = f"pp_data/ngrams_data_{n}_{125}.pth"
    tensor_data = load_data(file_path=filepath)
    X_tensor, t_tensor = tensor_data.tensors
    X_tensor, t_tensor = X_tensor.to(device), t_tensor.to(device)
    t_laptime = t_tensor[:, 1]
    t_postion = t_tensor[:, 57:77]
    total_rows = X_tensor.shape[0]
    train_count, val_count = int(total_rows * train_split), int(total_rows * (train_split+val_split))
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor[:train_count], t_laptime[:train_count], t_postion[:train_count])
    val_dataset = torch.utils.data.TensorDataset(X_tensor[train_count:val_count], t_laptime[train_count:val_count], t_postion[train_count:val_count])
    test_dataset = torch.utils.data.TensorDataset(X_tensor[val_count:], t_laptime[val_count:], t_postion[val_count:])
    if VERBOSE:
        print(f"train size: {len(train_dataset)} \t val size: {len(val_dataset)} \t test size: {len(test_dataset)}")
        print(f"train device:{train_dataset.tensors[0].device} \t val device:{val_dataset.tensors[0].device} \t test device:{test_dataset.tensors[0].device}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if VERBOSE:
        X, t_lt, t_p = next(iter(train_dataloader))
        print(f"X={X.shape} \t t_laptime={t_lt.shape} \t t_position={t_p.shape}")
        print(f"X device:{X.device} \t t_laptime device:{t_lt.device} \t t_position device:{t_p.device}")
    
    return train_dataloader, val_dataloader, test_dataloader

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

if __name__ == "__main__":

    file_path = "data/processed-data.csv"
    train_dataloader, val_dataloader, test_dataloader = get_data(n=3, batch_size=16, VERBOSE=True)

    model1 = ANN_MIMO_v2(input_num=2, input_size=125, hidden_dim=100, emb_dim=30, hidden_output_list=[5, 25], act_fn='relu', optimiser='adam', lr=0.0001)
    loss_model1 = Loss_Model_v1(optimiser='adam', lr=0.1)

    print(f"\n === Training model === \n")
    
    metrics = train_model(model=model1, loss_model=loss_model1, train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=10, verbose_every=100, VERBOSE=True)

    import json
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)


