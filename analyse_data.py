import json

with open('train_metrics_lstm_v0_1.json') as f:
  data = json.load(f)

print(len(data))

errors = 0

for obs in data:
    try:
       print(f"time: {obs['total_time']['seconds']}.{str(obs['total_time']['microseconds'])[:2]}\t"
             f"layers: {obs['hyperparameters']['num_layers']}\tbatchsize:{obs['hyperparameters']['batch_size']}\tn: {obs['hyperparameters']['n']}\t"
             f"laptime hidden: {obs['hyperparameters']['laptime_hidden_size']}\t"
             f"position hidden: {obs['hyperparameters']['position_hidden_size']}\t"
             f"lr:{obs['hyperparameters']['learning_rate']}\t"
             f"best train {max(obs['metrics']['position_train_acc'])*100}\tbest val {max(obs['metrics']['position_val_acc'])*100}")
    except TypeError:
       print(f"time: {obs['total_time']['seconds']}.{obs['total_time']['microseconds']}\t"
              f"layers: {obs['n']}\t"
              f"laptime hidden: {obs['hyperparameters']['laptime_hidden_size']} \t"
              f"position hidden: {obs['hyperparameters']['position_hidden_size']}")
    except Exception as e:
        errors += 1
        print(f"Error: {e}")

print(errors)