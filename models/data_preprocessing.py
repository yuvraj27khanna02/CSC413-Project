import pandas as pd
import torch


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

def get_data(file_path=str, n=int, data_dim=int):
    
    df = pd.read_csv(file_path).iloc[:, 1:]
    ngrams_data_df = generate_ngrams(df, n).dropna()

    ngrams_data_tensor = torch.tensor(ngrams_data_df.values, dtype=torch.float32)
    X_tensor, t_tensor = ngrams_data_tensor[:, :-data_dim], ngrams_data_tensor[:, -data_dim:]
    tensor_data = torch.utils.data.TensorDataset(X_tensor, t_tensor)
    
    return tensor_data

def load_data(file_path):
    try:
        tensor_data = torch.load(file_path)
        print(f'Data loaded successfully from {file_path}')
        print(f"length: {len(tensor_data)} \t shape: {tensor_data[0][0].size()} , {tensor_data[0][1].size()}")
        return tensor_data
    except Exception as e:
        print(f" \t ====== Error ====== \n{e}")
        return e

def save_data(preprocess_data_path=str, n=int, data_dim=int):
    """
    Saves preprocessed data as a PyTorch tensor to a specified file path.

    Args:
        preprocess_data_path (str): The file path of the preprocessed data.
        n (int): The value of 'n' for n-grams.
        data_dim (int): The dimension of the data.
    """
    file_path = f"ready_data/ngrams_data_{n}_{data_dim}.pth"
    tensor_data = get_data(file_path=preprocess_data_path, n=n, data_dim=data_dim)
    try:
        torch.save(tensor_data, file_path)
        print(f'Data saved successfully in {file_path}')
        return file_path
    except Exception as e:
        print(f" \t ====== Error ====== \n{e}")
        return e

if __name__ == "__main__":

    # from datetime import datetime
    
    PREPROCESS_DATA_PATH = "data/processed-data.csv"
    # print('saving data ...')
    
    # start_time = datetime.now()
    # prev_datetime = start_time
    
    # for i in range(1, 15):
    #     print(i)
    #     file_path = save_data(preprocess_data_path=PREPROCESS_DATA_PATH, n=i, data_dim=125)
        
    #     curr_time = datetime.now()
    #     print(f"Time \t total: {curr_time - start_time} \t last: {curr_time - prev_datetime}")
    #     prev_datetime = curr_time

    # print("data saved \nloading data ...")
    # n_ = int(input("Enter n value:"))
    # data_dim_ = int(input("Enter dim_data value:"))
    # file_path = f"pp_data/ngrams_data_{n_}_{data_dim_}.pth"
    # loaded_data = load_data(file_path=file_path)
    # print('data loaded')

    file_path = save_data(preprocess_data_path=PREPROCESS_DATA_PATH, n=3, data_dim=125)

