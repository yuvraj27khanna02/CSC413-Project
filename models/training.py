import pandas as pd
import torch
from arch import ANN_MIMO_v2

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

    return ngrams_df

def data_to_device(data:torch.utils.data.DataLoader, device:torch.device):
    for X, t_lt, t_p in data:
        X, t_lt, t_p = X.to(device), t_lt.to(device), t_p.to(device)

def print_df(df:pd.DataFrame):
    index_i = 0
    print('index \t value \t dtype \t column name')
    for col in df.columns:
        print(f"{index_i} \t| {round(df[col][0], 4)} \t| {df[col].dtype} \t| {col}")
        index_i += 1

def get_laptime_accuracy(dataset:torch.utils.data.TensorDataset, model:torch.nn.Module):
    total_accuracy = 0
    total_count = 0
    for (X, t_laptime, t_position) in dataset:
        y_laptime, y_position = model(X)
        accuracy = torch.mean(torch.abs(y_laptime - t_laptime))
        total_accuracy += accuracy
        total_count += 1
    return total_accuracy / total_count

def get_position_accuracy(dataset:torch.utils.data.TensorDataset, model:torch.nn.Module):
    total_correct = 0
    total_count = 0
    for (X, t_laptime, t_position) in dataset:
        y_laptime, y_position = model(X)
        pred_position = y_position >= 0.5
        total_correct += int(torch.sum(pred_position == t_position))
        total_count += t_position.shape[0]
    return total_correct / total_count

def get_data(file_path=str, n=3, train_split=0.6, val_split=0.2,
             batch_size=32,
             VERBOSE=False):

    df = pd.read_csv(file_path)
    ngrams_data_df = generate_ngrams(df, n)
    if VERBOSE:
        print_df(ngrams_data_df)
    
    # TODO: which dtype to use?
    ngrams_data_tensor = torch.tensor(ngrams_data_df.values, dtype=torch.float32)
   
    X_tensor, t_tensor = ngrams_data_tensor[:, :-125], ngrams_data_tensor[:, -125:]
    t_laptime = t_tensor[:, 1]
    t_postion = t_tensor[:, 57:77]
    total_rows = ngrams_data_tensor.shape[0]
    train_count, val_count = int(total_rows * train_split), int(total_rows * (train_split+val_split))
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor[:train_count], t_laptime[:train_count], t_postion[:train_count])
    val_dataset = torch.utils.data.TensorDataset(X_tensor[train_count:val_count], t_laptime[train_count:val_count], t_postion[train_count:val_count])
    test_dataset = torch.utils.data.TensorDataset(X_tensor[val_count:], t_laptime[val_count:], t_postion[val_count:])
    if VERBOSE:
        print(f"train: {len(train_dataset)} \t val: {len(val_dataset)} \t test: {len(test_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if VERBOSE:
        X, t_lt, t_p = next(iter(train_dataloader))
        print(f"X={X.shape} \t t_laptime={t_lt.shape} \t t_position={t_p.shape}")
    
    return train_dataloader, val_dataloader, test_dataloader

def train_model(model:ANN_MIMO_v2,
                train_dataloader:torch.utils.data.DataLoader,
                val_dataloader:torch.utils.data.DataLoader,
                num_epochs=10, 
                verbose_every=10,
                criterion_laptime=torch.nn.MSELoss(),
                criterion_position=torch.nn.CrossEntropyLoss(),
                VERBOSE=False):
    
    # model.to(model.get_device())
    # data_to_device(train_dataloader, model.get_device())
    # data_to_device(val_dataloader, model.get_device())

    train_loss_sum_list = []
    train_loss_laptime_list = []
    train_loss_position_list = []
    train_acc_laptime_list = []
    train_acc_position_list = []
    val_acc_laptime_list = []
    val_acc_position_list = []
    epoch_list = []
    itr_list = []

    optimiser = model.get_optimiser()
    iter_count = 0

    for epoch in range(num_epochs):
        for (X, t_laptime, t_position) in train_dataloader:

            t_laptime = t_laptime.view(-1, 1)

            model.train()

            y_laptime, y_position = model(X)
            loss_laptime = criterion_laptime(y_laptime, t_laptime)
            
            # TODO: If error line below try t_position.softmax(dim=1)
            loss_position = criterion_position(y_position, t_position)

            # TODO: test different loss weights?  If result not good then try backward pass separately?
            loss = loss_laptime + loss_position

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

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

            if (VERBOSE) and (iter_count % verbose_every == 0):
                print(f"epoch:{epoch} \t itr:{iter_count} \t batch size:{y_laptime.shape[0]} \t loss:{loss.item()}"
                      f"\ntrain laptime loss: {loss_laptime.item()} \t train laptime accuracy: {train_laptime_accuracy} \t val laptime accuracy: {val_laptime_accuracy}"
                      f"\ntrain position loss: {loss_position.item()} \t train position accuracy: {train_position_accuracy} \t val position accuracy: {val_position_accuracy}\n")

            iter_count += 1
    
    return {
        'optimiser': optimiser,
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
    train_dataloader, val_dataloader, test_dataloader = get_data(file_path, n=3, VERBOSE=True)

    model1 = ANN_MIMO_v2(input_num=2, input_size=125, hidden_dim=100, emb_dim=30, hidden_output_list=[5, 25], act_fn='relu', optimiser='adam', lr=0.01)

    print(f"\n === Training model === \n")
    
    metrics = train_model(model=model1, train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=10, verbose_every=100, VERBOSE=True)

    import json
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)


