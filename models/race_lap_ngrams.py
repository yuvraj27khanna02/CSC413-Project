import pandas as pd
import numpy as np
import torch

# Change this to "data/processed-data-split-laptime.csv" to use the split laptime data
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

    small = False

    def __init__(self, n, processed_data_path = PROCESSED_DATA_PATH, small = False):
        self.small = small
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
                if lap_numbers.to_list() == list(range(min_lap_number, min_lap_number + n)):
                    self.ngram_indices.append(i)

    def split_by_year(self):
        train_years = [2019] if self.small else [2019, 2020, 2021]
        val_years = [2022]
        test_years = [2023]

        ngram_subset = self.orig_df.iloc[self.ngram_indices]
        
        self.train_indices = ngram_subset[ngram_subset['Year'].isin(train_years)].index.tolist()
        self.val_indices = ngram_subset[ngram_subset['Year'].isin(val_years)].index.tolist()
        self.test_indices = ngram_subset[ngram_subset['Year'].isin(test_years)].index.tolist()

    def split_by_proportion(self, train_proportion = 0.6, val_proportion = 0.2):
        shuffled_indices = np.random.permutation(self.ngram_indices)

        train_split = int(len(self.ngram_indices) * train_proportion)
        val_split = int(len(self.ngram_indices) * val_proportion)

        self.train_indices = shuffled_indices[:train_split].tolist()
        self.val_indices = shuffled_indices[train_split:train_split + val_split].tolist()
        self.test_indices = shuffled_indices[train_split + val_split:].tolist()

    def get_train_tensors(self):
        return self.get_tensors(self.train_indices)
    
    def get_val_tensors(self):
        return self.get_tensors(self.val_indices)
    
    def get_test_tensors(self):
        return self.get_tensors(self.test_indices)

    def get_tensors(self, indices):
        expected_ngram_size = self.n * self.data_dim

        ngrams = [self.df.iloc[i:i+self.n] for i in indices]
        ngrams_array = [ngram.values.flatten() for ngram in ngrams]
        ngrams_array = [ngram for ngram in ngrams_array if len(ngram) == expected_ngram_size]

        data_tensor = torch.tensor(np.array(ngrams_array), dtype=torch.float32)

        X_tensor, t_tensor = data_tensor[:, :-self.data_dim], data_tensor[:, -self.data_dim:]
        t_laptime = t_tensor[:, 1]
        t_position = t_tensor[:, 57:77]

        return X_tensor, t_laptime, t_position



if __name__ == "__main__":
    three_gram = RaceLapNgrams(20)
    print(three_gram.data_dim)

