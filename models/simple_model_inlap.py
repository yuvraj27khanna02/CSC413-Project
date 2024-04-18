from race_lap_ngrams import RaceLapNgrams
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

n = 3

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
    
    
def train_laptime_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
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
            predictions.extend(outputs.squeeze().tolist())
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
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')

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
    race_lap_ngrams = RaceLapNgrams(n=n)
    race_lap_ngrams.split_by_year()
    
    train_X, train_laptime, train_position = race_lap_ngrams.get_train_tensors()
    val_X, val_laptime, val_position = race_lap_ngrams.get_val_tensors()
    test_X, test_laptime, test_position = race_lap_ngrams.get_test_tensors()

    # Get inlap information
    inlap_index = race_lap_ngrams.indices['InLap']
    train_inlap = train_X[:, :, inlap_index]
    val_inlap = val_X[:, :, inlap_index]
    test_inlap = test_X[:, :, inlap_index]

    # Laptime
    laptime_index = race_lap_ngrams.indices['LapTime']
    train_X_laptime = torch.cat((train_X[:, :, laptime_index], train_inlap), dim=1)
    val_X_laptime = torch.cat((val_X[:, :, laptime_index], val_inlap), dim=1)
    test_X_laptime = torch.cat((test_X[:, :, laptime_index], test_inlap), dim=1)

    train_laptime_loader = DataLoader(TensorDataset(train_X_laptime, train_laptime), batch_size=32, shuffle=True)
    val_laptime_loader = DataLoader(TensorDataset(val_X_laptime, val_laptime), batch_size=32)
    test_laptime_loader = DataLoader(TensorDataset(test_X_laptime, test_laptime), batch_size=32)

    laptime_mlp = LapTime_MLP(input_dim=(n-1)*2)
    train_losses, test_losses = train_laptime_model(laptime_mlp, train_laptime_loader, val_laptime_loader, num_epochs=100)
    mae, mse, rmse = calculate_laptime_metrics(laptime_mlp, test_laptime_loader)
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    # Position
    pos_start_idx = race_lap_ngrams.indices['Position_1.0']
    pos_end_idx = race_lap_ngrams.indices['Position_20.0'] + 1

    train_X_position = torch.cat((train_X[:, :, pos_start_idx:pos_end_idx], train_inlap.reshape(-1, (n-1), 1)), dim=-1).reshape(-1, (n-1) * 21)
    val_X_position = torch.cat((val_X[:, :, pos_start_idx:pos_end_idx], val_inlap.reshape(-1, (n-1), 1)), dim=-1).reshape(-1, (n-1) * 21)
    test_X_position = torch.cat((test_X[:, :, pos_start_idx:pos_end_idx], test_inlap.reshape(-1, (n-1), 1)), dim=-1).reshape(-1, (n-1) * 21)

    train_position_loader = DataLoader(TensorDataset(train_X_position, train_position), batch_size=32, shuffle=True)
    val_position_loader = DataLoader(TensorDataset(val_X_position, val_position), batch_size=32)
    test_position_loader = DataLoader(TensorDataset(test_X_position, test_position), batch_size=32)

    position_mlp = Position_MLP(n=n-1, input_dim=21)
    train_position_model(position_mlp, train_position_loader, val_position_loader, num_epochs=25)
    test_accuracy = evaluate_position_model(position_mlp, test_position_loader)
    print(f'Position MLP Test Accuracy: {test_accuracy:.4f}')
