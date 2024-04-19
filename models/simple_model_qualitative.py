from race_lap_ngrams import RaceLapNgrams
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import numpy as np

class LapTime_MLP(nn.Module):
    def __init__(self, input_dim):
        super(LapTime_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class Position_MLP(nn.Module):
    def __init__(self, n, input_dim = 20):
        super(Position_MLP, self).__init__()

        self.fc1 = nn.Linear(n * input_dim, 50)
        self.fc2 = nn.Linear(50, 20)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def train_laptime_model(model, train_loader, val_loader, num_epochs=50, lr=0.0005):
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Tracking loss for plotting
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        
        for data, targets in train_loader:
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate on the test set
        val_loss = evaluate_laptime_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        # Print training progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses

def evaluate_laptime_model(model, loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item() * data.size(0)
    
    return total_loss / len(loader.dataset)

def calculate_laptime_metrics(model, loader):
    model.eval()  # Set model to evaluation mode
    actuals, predictions = [], []
    
    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data)
            predictions.extend(outputs.tolist())
            actuals.extend(targets.tolist())
    
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    return mae, mse, rmse

def train_position_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Tracking loss for plotting
    train_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        
        for data, targets in train_loader:
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Print training progress
        if (epoch+1) % 10 == 0:
            train_acc = evaluate_position_model(model, train_loader)
            val_acc = evaluate_position_model(model, val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Loss: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')

    return train_losses#, val_losses


def evaluate_position_model(model, loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            output = model(data)

            _, predicted = torch.max(output, 1)
            _, targets_labels = torch.max(targets, 1)  # Convert one-hot to integer labels

            correct += (predicted == targets_labels).sum().item()
            total += targets.size(0)

    
    return correct / total
    
if __name__ == "__main__":
    for n in [5]:
    #     race_lap_ngrams = RaceLapNgrams(n=n, hammer_time=False)
    #     race_lap_ngrams.split_by_year()
        
    #     train_X, train_laptime, train_position = race_lap_ngrams.get_train_tensors()
    #     val_X, val_laptime, val_position = race_lap_ngrams.get_val_tensors()
    #     test_X, test_laptime, test_position = race_lap_ngrams.get_test_tensors()

    #     # Laptime
    #     laptime_index = race_lap_ngrams.indices['LapTime']
    #     train_X_laptime = train_X[:, :, laptime_index]
    #     val_X_laptime = val_X[:, :, laptime_index]
    #     test_X_laptime = test_X[:, :, laptime_index]

    #     train_laptime_loader = DataLoader(TensorDataset(train_X_laptime, train_laptime), batch_size=32, shuffle=True)
    #     val_laptime_loader = DataLoader(TensorDataset(val_X_laptime, val_laptime), batch_size=32)
    #     test_laptime_loader = DataLoader(TensorDataset(test_X_laptime, test_laptime), batch_size=32)

    #     laptime_mlp = LapTime_MLP(input_dim=n-1)
    #     train_losses, test_losses = train_laptime_model(laptime_mlp, train_laptime_loader, val_laptime_loader, num_epochs=100)
    #     mae, mse, rmse = calculate_laptime_metrics(laptime_mlp, test_laptime_loader)
    #     # print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    #     # Position
    #     pos_start_idx = race_lap_ngrams.indices['Position_1.0']
    #     pos_end_idx = race_lap_ngrams.indices['Position_20.0'] + 1
    #     train_X_position = train_X[:, :, pos_start_idx:pos_end_idx].reshape(-1, (n-1) * 20)
    #     val_X_position = val_X[:, :, pos_start_idx:pos_end_idx].reshape(-1, (n-1) * 20)
    #     test_X_position = test_X[:, :, pos_start_idx:pos_end_idx].reshape(-1, (n-1) * 20)

    #     train_position_loader = DataLoader(TensorDataset(train_X_position, train_position), batch_size=32, shuffle=True)
    #     val_position_loader = DataLoader(TensorDataset(val_X_position, val_position), batch_size=32)
    #     test_position_loader = DataLoader(TensorDataset(test_X_position, test_position), batch_size=32)

    #     position_mlp = Position_MLP(n=n-1)
    #     train_position_model(position_mlp, train_position_loader, val_position_loader, num_epochs=10)
    #     test_accuracy = evaluate_position_model(position_mlp, test_position_loader)
    #     # print(f'Position MLP Test Accuracy: {test_accuracy:.4f}')

    #     print(f"n={n}: Position accuracy: {test_accuracy}, Laptime error: MAE {mae}, MSE {mse}, RMSE {rmse}")

    #     torch.save(laptime_mlp, f"models/laptime_mlp_{n}.pt")
    #     torch.save(position_mlp, f"models/position_mlp_{n}.pt")
        laptime_mlp = torch.load(f"models/laptime_mlp_{n}.pt")
        position_mlp = torch.load(f"models/position_mlp_{n}.pt")
        laptime_mlp.eval()
        position_mlp.eval()

        # Check scaling
        print(laptime_mlp(torch.tensor([10000.0, 9900.0, 9800.0, 9700.0]))) # 9629.3438
        print(laptime_mlp(torch.tensor([1000.0, 990.0, 980.0, 970.0]))) # 983.0688

        # Decreasing
        print(laptime_mlp(torch.tensor([91000.0, 90800.0, 90600.0, 90400.0]))) # 90289.1016

        # Increasing
        print(laptime_mlp(torch.tensor([90400.0, 90600.0, 90800.0, 91000.0]))) # 91290.4844

        # Bouncing around
        print(laptime_mlp(torch.tensor([90600.0, 90400.0, 91000.0, 90800.0]))) # 91066.2109

        # Position
        def positions_tensor(positions):
            tensors = []
            for position in positions:
                pos_tensor = torch.zeros(20)
                pos_tensor[position-1] = 1
                tensors.append(pos_tensor)
            
            return torch.cat(tensors)
    
        
        import matplotlib.pyplot as plt

        # Store tensors in numpy arrays
        stable = torch.softmax(position_mlp(positions_tensor([5, 5, 5, 5])), dim=0).detach().numpy()
        climbing = torch.softmax(position_mlp(positions_tensor([7, 7, 6, 5])), dim=0).detach().numpy()
        falling = torch.softmax(position_mlp(positions_tensor([1, 1, 2, 3])), dim=0).detach().numpy()
        extreme = torch.softmax(position_mlp(positions_tensor([18, 1, 5, 10])), dim=0).detach().numpy()

        # Create histogram
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Next Position Probabilities', fontsize=16)

        # Plot histograms for each tensor
        x = np.arange(1, 21)

        axs[0].bar(x, stable)
        axs[0].set_title('Previous Positions Stable: 5, 5, 5, 5')
        axs[1].bar(x, climbing)
        axs[1].set_title('Previous Positions Climbing: 7, 7, 6, 5')
        axs[2].bar(x, falling)
        axs[2].set_title('Previous Positions Falling: 1, 1, 2, 3')
        axs[3].bar(x, extreme)
        axs[3].set_title('Previous Positions extreme: 18, 1, 5, 10')

        # Set x-axis labels
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels(x)

        # Save figure
        plt.savefig('position_histogram.png')