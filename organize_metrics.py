import json

def get_metrics(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

        # data is an array
        # each element is a model represented as a dictionary
        # each dictionary has metrics with laptime_val_mae, laptime_val_mse, position_val_acc
        # the last laptime_val_mae is the model's metric
        # get the model with the lowest laptime_val_mae
        # return the model's metrics and hyperparameters

        best_model = None
        best_mae = float('inf')
        for model in data:
            mae = model['metrics']['laptime_val_mae'][-1]
            if mae < best_mae:
                best_model = model
                best_mae = mae

        model_info = best_model['model_info']
        hyperparameters = best_model['hyperparameters']
        model_info.update(hyperparameters)
        metrics = best_model['metrics']
        mae = metrics['laptime_val_mae'][-1]
        mse = metrics['laptime_val_mse'][-1]
        acc = metrics['position_val_acc'][-1]

        return model_info, mae, mse, acc

if __name__ == '__main__':
    best_mlp, mlp_mae, mlp_mse, mlp_acc = get_metrics('mlp_metrics_3.json')
    print(best_mlp, mlp_mae, mlp_mse, mlp_acc)

    best_rnn, rnn_mae, rnn_mse, rnn_acc = get_metrics('rnn_metrics_3.json')
    print(best_rnn, rnn_mae, rnn_mse, rnn_acc)