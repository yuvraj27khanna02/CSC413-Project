import pandas as pd
import numpy as np
import torch


PROCESSED_DATA_PATH = "data/processed-data.csv"


class RaceLapNgrams:
    orig_df: pd.DataFrame
    df: pd.DataFrame

    n: int
    data_dim = int

    ngram_indices = []

    train_indices = []
    val_indices = []
    test_indices = []

    def __init__(self, n, processed_data_path = PROCESSED_DATA_PATH):
        self.orig_df = pd.read_csv(processed_data_path).iloc[:, 1:]
        self.orig_df.sort_values(
            by=['Year', 'Event_Orig', 'Driver_Orig', 'LapNumber'],
            inplace=True,
            ignore_index=True)

        self.df = self.orig_df.drop(columns=['Event_Orig', 'Driver_Orig'])

        self._get_ngram_indices(n)

        self.n = n
        self.data_dim = len(self.df.columns)


    def _get_ngram_indices(self, n):
        for _, group in self.orig_df.groupby(['Year', 'Event_Orig', 'Driver_Orig']):
            indices = group.index.to_list()
            for i in indices[: len(indices) - n  + 1]:
                potential_ngram = self.orig_df.iloc[i: i + n]

                lap_numbers = potential_ngram['LapNumber']
                min_lap_number = lap_numbers.min()
                if lap_numbers.tolist() == list(range(min_lap_number, min_lap_number + n)):
                    self.ngram_indices.append(i)

    def split_by_year(self):
        train_years = [2019, 2020, 2021]
        val_years = [2022]
        test_years = [2023]

        ngram_subset = self.orig_df.iloc[self.ngram_indices]
        
        self.train_indices = ngram_subset[ngram_subset['Year'].isin(train_years)].index.to_list()
        self.val_indices = ngram_subset[ngram_subset['Year'].isin(val_years)].index.to_list()
        self.test_indices = ngram_subset[ngram_subset['Year'].isin(test_years)].index.to_list()

    def split_by_proportion(self, train_proportion = 0.6, val_proportion = 0.2):
        shuffled_indices = np.random.permutation(self.ngram_indices)

        train_split = int(len(self.ngram_indices) * train_proportion)
        val_split = int(len(self.ngram_indices) * val_proportion)

        self.train_indices = shuffled_indices[:train_split]
        self.val_indices = shuffled_indices[train_split:train_split + val_split]
        self.test_indices = shuffled_indices[train_split + val_split:]

    def get_tensors(self, indices):
        indices = indices.tolist()

        ngrams = [self.df.iloc[i:i+self.n] for i in indices]
        data_tensor = torch.tensor(np.array([ngram.values.flatten() for ngram in ngrams]), dtype=torch.float32)

        X_tensor, t_tensor = data_tensor[:, :-self.data_dim], data_tensor[:, -self.data_dim:]
        t_laptime = t_tensor[:, 1]
        t_position = t_tensor[:, 57:77]

        return X_tensor, t_laptime, t_position



if __name__ == "__main__":
    three_gram = RaceLapNgrams(20)
    print(three_gram.data_dim)

