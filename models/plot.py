import json
import matplotlib.pyplot as plt

# Load the metrics from the JSON file
def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def plot_accuracy(metrics, output_type):
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        hyperparameters = metric['hyperparameters']
        if output_type == 'laptime':
            plt.plot(metric['metrics']['epoch_list'], metric['metrics']['train_acc_laptime'], label=f"train_acc_{hyperparameters}")
            plt.plot(metric['metrics']['epoch_list'], metric['metrics']['val_acc_laptime'], label=f"val_acc_{hyperparameters}")
        elif output_type == 'position':
            plt.plot(metric['metrics']['epoch_list'], metric['metrics']['train_acc_position'], label=f"train_acc_{hyperparameters}")
            plt.plot(metric['metrics']['epoch_list'], metric['metrics']['val_acc_position'], label=f"val_acc_{hyperparameters}")

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title(f'{output_type.capitalize()} Accuracy')
    plt.legend()
    plt.show()


def plot_loss(metrics, output_type):
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        hyperparameters = metric['hyperparameters']
        if output_type == 'laptime':
            plt.plot(metric['metrics']['epoch_list'], metric['metrics']['train_loss_laptime'], label=f"train_loss_{hyperparameters}")
        elif output_type == 'position':
            plt.plot(metric['metrics']['epoch_list'], metric['metrics']['train_loss_position'], label=f"train_loss_{hyperparameters}")
        plt.plot(metric['metrics']['epoch_list'], metric['metrics']['train_loss_laptime'], label=f"train_loss_{hyperparameters}")

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    metrics = load_metrics('train_metrics.json')
    plot_accuracy(metrics, output_type='laptime')
    plot_accuracy(metrics, output_type='position')

    plot_loss(metrics, output_type='laptime')
    plot_loss(metrics, output_type='position')
