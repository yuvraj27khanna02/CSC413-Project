import pandas as pd
import torch

FILE_PATH = "data/bahrain-2022.csv"
bahrain_22_data = pd.read_csv(FILE_PATH)

def process_data_1(data, cols_to_one_hot=["Driver", "Compound", "Team", "Position"]):

    temp_data = pd.get_dummies(data, columns=cols_to_one_hot, dtype=int)
    temp_data.to_csv('data/bhr_22_process.csv')
    temp_data = temp_data.iloc[:, 1:]
    columns = temp_data.columns
    tensor = torch.tensor(temp_data.values, dtype=torch.float32)

    return tensor, columns

pp_data, cols = process_data_1(bahrain_22_data)

# print(pp_data.size())

# for i, col in enumerate(cols):
#     print(f"{col}: {pp_data[0][i]}")

# print(pp_data[0])


