import json

def collate_data(filenames:list[str], final_filename:str):
    data = []
    i = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            data.extend(json.load(f))
            i += 1
    with open(final_filename, 'w') as f:
        json.dump(data, f)
    print('done !')


if __name__ == '__main__':
    final_filename = 'final_metrics.json'

    filenames = []

    for n in [3, 7, 10, 15]:
        filenames.append(f'mlp_metrics_{n}.json')
        filenames.append(f'rnn_metrics_{n}.json')
        filenames.append(f'lstm_metrics_{n}.json')

    collate_data(filenames, final_filename)