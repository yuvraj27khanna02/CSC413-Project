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

class Laptime_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Laptime_RNN, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU()
        )
        self.rnn = nn.RNN(input_size=16, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_n = nn.Sequential(
            nn.Linear(hidden_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, X):
        x = self.fc_1(X)
        out_, hidden_ = self.rnn(x)
        out = out_[:, -1, :]
        out = self.fc_n(out)
        return out

class Position_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(Position_RNN, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU()
        )
        self.rnn = nn.RNN(input_size=16, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_n = nn.Sequential(
            nn.Linear(hidden_dim, 10),
            nn.ReLU(),
            nn.Linear(10, out_dim)
        )
    
    def forward(self, X):
        x = self.fc_1(X)
        out_, hidden_ = self.rnn(x)
        out = out_[:, -1, :]
        out = self.fc_n(out)
        return out
    
    
def train_laptime_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=0.0, momentum=0.9):
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Tracking loss for plotting
    train_losses, val_losses = [], []
    val_mae_list = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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
        val_loss, val_mae = evaluate_laptime_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_mae_list.append(val_mae)
        
        # Print training progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        scheduler.step()

    return train_losses, val_losses

def evaluate_laptime_model(model, loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    tae = 0.0
    
    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item() * data.size(0)
            tae += torch.sum(torch.abs(outputs.squeeze() - targets)).item()
    
    return total_loss / len(loader.dataset), tae / len(loader.dataset)

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

def train_position_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=0.0, momentum=0.9):
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Tracking loss for plotting
    train_losses = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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
        
        scheduler.step()

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

    for weight_decay in [0.0, 1e-1, 1e-3, 1e-5]:
        for momentum in [0.9, 0.99]:
            print("\t","="*100)
            print(f"Weight Decay: {weight_decay}, Momentum: {momentum}")
            # Laptime
            laptime_dim = 1
            laptime_index = race_lap_ngrams.laptime_index
            train_X_laptime = train_X[:, :, laptime_index].view(-1, n-1, laptime_dim)
            val_X_laptime = val_X[:, :, laptime_index].view(-1, n-1, laptime_dim)
            test_X_laptime = test_X[:, :, laptime_index].view(-1, n-1, laptime_dim)

            train_laptime_loader = DataLoader(TensorDataset(train_X_laptime, train_laptime), batch_size=32, shuffle=True)
            val_laptime_loader = DataLoader(TensorDataset(val_X_laptime, val_laptime), batch_size=32)
            test_laptime_loader = DataLoader(TensorDataset(test_X_laptime, test_laptime), batch_size=32)

            # laptime_model = LapTime_MLP(input_dim=n-1)
            laptime_model = Laptime_RNN(input_dim=laptime_dim, hidden_dim=32, num_layers=1)
            print(laptime_model)
            train_losses, test_losses = train_laptime_model(laptime_model, train_laptime_loader, val_laptime_loader, 
                                                            num_epochs=400, lr=5e-4, weight_decay=weight_decay, momentum=momentum)
            mae, mse, rmse = calculate_laptime_metrics(laptime_model, test_laptime_loader)
            print(f"Laptime RNN Test MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

            # Position
            position_dim = 20
            pos_start_idx, pos_end_idx = (race_lap_ngrams.position_start_index, race_lap_ngrams.position_start_index + 20) 
            train_X_position = train_X[:, :, pos_start_idx:pos_end_idx].reshape(-1, n-1, position_dim)
            val_X_position = val_X[:, :, pos_start_idx:pos_end_idx].reshape(-1, n-1, position_dim)
            test_X_position = test_X[:, :, pos_start_idx:pos_end_idx].reshape(-1, n-1, position_dim)

            train_position_loader = DataLoader(TensorDataset(train_X_position, train_position), batch_size=32, shuffle=True)
            val_position_loader = DataLoader(TensorDataset(val_X_position, val_position), batch_size=32)
            test_position_loader = DataLoader(TensorDataset(test_X_position, test_position), batch_size=32)

            # position_model = Position_MLP(n=n-1)
            position_model = Position_RNN(input_dim=20, hidden_dim=32, num_layers=1, out_dim=20)
            print(position_model)
            train_losses = train_position_model(position_model, train_position_loader, val_position_loader, 
                                                num_epochs=400, lr=5e-4, weight_decay=weight_decay, momentum=momentum)
            test_accuracy = evaluate_position_model(position_model, test_position_loader)
            print(f'Position RNN Test Accuracy: {test_accuracy:.4f}')

