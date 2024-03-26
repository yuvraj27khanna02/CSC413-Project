import json

with open('train_metrics_1.json') as f:
  data = json.load(f)

print(len(data))

errors = 0

for obs in data:
    try:
       print(f"time: {obs['total_time']['seconds']}.{str(obs['total_time']['microseconds'])[:2]}\t"
             f"layers: {obs['hyperparameters']['num_layers']}\t n: {obs['hyperparameters']['n']}\t"
             f"laptime hidden: {obs['hyperparameters']['laptime_hidden_size']}\t"
             f"position hidden: {obs['hyperparameters']['position_hidden_size']}\t"
             f"lr:{obs['hyperparameters']['learning_rate']}\t"
             f"best {max(obs['metrics']['val_acc_position'])*100}")
    except TypeError:
       print(f"time: {obs['total_time']['seconds']}.{obs['total_time']['microseconds']}\t"
              f"layers: {obs['n']}\t"
              f"laptime hidden: {obs['hyperparameters']['laptime_hidden_size']} \t"
              f"position hidden: {obs['hyperparameters']['position_hidden_size']}")
    except Exception as e:
        errors += 1

print(errors)