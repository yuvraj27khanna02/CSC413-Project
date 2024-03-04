import pandas as pd
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_data(file_path=str, continous_columns=list, categorical_columns=list, target_columns=str):
    file_df = pd.read_csv(file_path)
    pre_processed_df = pd.DataFrame({})

    for continous_variable in continous_columns:
        pre_processed_df[continous_variable] = file_df[continous_variable].astype(float)
    
    for categorical_variable in categorical_columns:
        pre_processed_df = pd.concat([pre_processed_df, pd.get_dummies(file_df[categorical_variable])], axis=1)
    
    for target_variable in target_columns:
        pre_processed_df[target_variable] = file_df[target_variable].astype(float)
    return pre_processed_df


def process_laptime_v1(laptime=str, VERBOSE=False):
    try:
        process_1 = str(laptime).split(':')
        processed_laptime = [int(process_1[1]), float(process_1[2]), round(60*int(process_1[1]) + float(process_1[2]), 7)]
    except IndexError as e:
        print(f'race ended for driver')
        processed_laptime = [0, 0, 0]
    except Exception as e:
        print(f'ERROR: {e}')
    if VERBOSE:
        print(f'laptime: {laptime} \t processed: {processed_laptime}')
    return processed_laptime


def preprocess_data_v1(file_path=str):
    file_df = pd.read_csv(file_path)
    laptime_df = pd.DataFrame([process_laptime_v1(laptime, VERBOSE=True) for laptime in file_df['LapTime']], columns=['LapTime_Minutes', 'LapTime_Seconds', 'LapTime_Total'])

    return laptime_df


FILE_PATH = 'data/bahrain-2022.csv'

pp_data = preprocess_data_v1(FILE_PATH)

print(pp_data.head())

