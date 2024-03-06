import pandas as pd
import torch


def process_laptime_v1(laptime=str, VERBOSE=False):
    try:
        process_1 = str(laptime).split(':')
        processed_laptime = [int(process_1[1]), float(process_1[2]), round(60*int(process_1[1]) + float(process_1[2]), 7)]
        processed_laptime_milliseconds = int(processed_laptime[2]*1000)
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

