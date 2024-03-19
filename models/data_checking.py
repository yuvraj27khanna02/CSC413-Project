import torch
import pandas as pd

def _print_df(df:pd.DataFrame):
    print(df.iloc[0])
    print('='*80)
    index_i = 0
    print('index \t | dtype  \t | column name')
    for col in df.columns:
        if 'Unnamed' in col:
            continue
        print(f"{index_i} \t| {df[col].dtype} \t| {col}")
        index_i += 1


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




if __name__ == "__main__":

    file_path = 'data/processed-data.csv'
    df = pd.read_csv(file_path)

    print(df.head())
    print(df.info())

    ngrams_3 = generate_ngrams(df, 3)

    # ngrams_3.to_csv('ngrams3.csv')

    ngrams_3 = ngrams_3.dropna()

    _print_df(ngrams_3)

    tensor_ngrams_3 = torch.tensor(ngrams_3.values, dtype=torch.float32)
    
    tensor_bool = tensor_ngrams_3.isnan()
    true_indices = torch.nonzero(tensor_bool)

    print(f"tensor_ngrams_3: {tensor_ngrams_3.size()} \t tensor_bool: {tensor_bool.size()} \t true_indices: {true_indices.size()}")

    for row_num, col_num in true_indices:
        print(f'row: {row_num.item()} \t col: {col_num.item()} \t col name: {ngrams_3.columns[col_num.item()]} \t value: {ngrams_3.iloc[row_num.item(), col_num.item()]}')
