import torch
import torch.nn as nn
from torchsummary import summary
from torchsummary import summary
import sys
import io
import os

os.environ['device'] = 'cpu'

def _get_act_fn(act_fn=str):
    """ act_fn list: ['relu', 'sig', 'tanh']
    """
    if act_fn == 'relu':
        return torch.nn.ReLU()
    if act_fn == 'sig':
        return torch.nn.Sigmoid()
    if act_fn == 'tanh':
        return torch.nn.Tanh()

def get_model_summary(model, batch_size=1):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        summary(model=model, input_size=(model.get_inputsize(),), batch_size=batch_size)
    except Exception as e:
        print(f'\t Error : {e}')
    a = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return a

def write_to_file(file_name, content):
    if os.path.isfile(file_name):
        with open(file_name, 'a') as f:
            f.write(content)
            f.close()
    else:
        with open(file_name, 'w') as f:
            f.write(content)
            f.close()

class Loss_Model_v1(torch.nn.Module):
    """
    Loss_Model_v1 class represents a custom loss model for a specific task.

    This model takes two inputs, `loss_laptime` and `loss_position`, and applies linear transformations
    to them using two linear layers (`w1` and `w2`). The output is the sum of the transformed inputs.

    Attributes:
        w1 (torch.nn.Linear): Linear layer for loss_laptime.
        w2 (torch.nn.Linear): Linear layer for loss_position.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, dtype=torch.float))
        self.w2 = torch.nn.Parameter(torch.tensor(5000.0, requires_grad=True, dtype=torch.float))

    def forward(self, loss_laptime, loss_position):
        """ Forward pass of the Loss_Model_v1.

        Args:
            loss_laptime (torch.Tensor): Input tensor 0D representing the loss for laptime.
            loss_position (torch.Tensor): Input tensor 0D representing the loss for position.
        """
        out = (self.w1 * loss_laptime) + (self.w2 * loss_position)
        return out
    
    def get_optimiser(self):
        return self.optimiser

class MLP_BC_v1(torch.nn.Module):
    """ Multilayer Perceptron for Binary Classification version 1

    """

    def __init__(self, input_size=int, hidden_size=int,  act_fn=str) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        x = self.act_fn(self.fc_1(input))
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.input_size
    
    def get_inputsize(self):
        return self.input_size

class MLP_regression_v1(torch.nn.Module):
    """ MultiLayer Perceptron for Regression version 1
    """

    def __init__(self, hidden_size=int, act_fn=str, data_dim=int, input_num=int) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(data_dim, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size*input_num, 1)
        self.act_fn = _get_act_fn(act_fn)
        self.data_dim = data_dim
    
    def forward(self, input):
        input_list = torch.split(input, self.data_dim, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in input_list], dim=1)
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.data_dim

class MLP_MC_v1(torch.nn.Module):
    """ Multilayer Perceptron for Multiclass Classification version 1
    """

    def __init__(self, input_num=int, hidden_size=int, output_classes=int, act_fn=str, data_dim=int) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(data_dim, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size*input_num, output_classes)
        self.act_fn = _get_act_fn(act_fn)
        self.data_dim = data_dim
    
    def forward(self, input):
        input_list = torch.split(input, self.data_dim, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in input_list], 1)
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.data_dim

class ANN_emb_v1(torch.nn.Module):
    def __init__(self, input_size=int, input_num=int, hidden_size=int, num_layers=int, emb_size=int, act_fn=str) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_num = input_num
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc_n = torch.nn.Linear(hidden_size*input_num, emb_size)

        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input_x):
        x = self.act_fn(self.fc_1(input_x))
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.input_size*self.input_num

class ANN_regression_v1(torch.nn.Module):
    """ Artificial Neural Network for Regression version 1
    
    Attributes:
        fc_1 (torch.nn.Linear): The first fully connected layer.
        fc_n (torch.nn.Linear): The second fully connected layer.
        act_fn (function): The activation function.
    
    Args:
        input_num (int): The number of input features.
        input_dim (int): The dimension of the input features.
        hidden_size (int): The number of units in the hidden layer.
        act_fn (str): The activation function to be used.
    """

    def __init__(self, input_size=int, input_num=int, num_layers=int, hidden_size=int, act_fn=str) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_num = input_num
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_2 = torch.nn.Linear(hidden_size*input_num, hidden_size)
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc_n = torch.nn.Linear(hidden_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        """Forward pass of the neural network.
        
        Args:
            input (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """

        x = torch.split(input, self.input_size, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in x], 1)
        x = self.act_fn(self.fc_2(x))
        for fc_layer in self.fc_layers:
            x = self.act_fn(fc_layer(x))
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.input_size*self.input_num

class ANN_regression_v2(torch.nn.Module):
    """MultiLayer Perceptron for Regression version 2 with tapering hidden dimensions

    Args:
            input_size (int): The size of the input features.
            hidden_sizes (List[int]): A list of hidden layer sizes. The number of hidden layers will be determined by the length of this list.
            act_fn (str): The activation function to use in the hidden layers.
    """

    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_sizes=list, act_fn=str) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_num = input_num
        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.fc_layers = torch.nn.ModuleList()
        prev_size = emb_size*input_num
        for size in hidden_sizes:
            self.fc_layers.append(torch.nn.Linear(prev_size, size))
            prev_size = size
        self.fc_n = torch.nn.Linear(prev_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        """
        Performs forward pass through the ANN_regression_v2 model.
        
        Args:
            input (torch.Tensor): The input tensor.
        """
        x = torch.split(input, self.input_size, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in x], 1)
        for fc_layer in self.fc_layers:
            x = self.act_fn(fc_layer(x))
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.input_size*self.input_num

class ANN_MC_v1(torch.nn.Module):
    """
    A class representing a version 1 of the ANN_MC architecture.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden layer.
        act_fn (str): The activation function to be used.

    Attributes:
        fc_1 (torch.nn.Linear): The first fully connected layer.
        fc_n (torch.nn.Linear): The final fully connected layer.
        act_fn (function): The activation function.

    """

    def __init__(self, input_size=int, input_num=int, hidden_size=int, num_layers=int, act_fn=str, output_classes=int) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_num = input_num
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_2 = torch.nn.Linear(hidden_size*input_num, hidden_size)
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc_n = torch.nn.Linear(hidden_size, output_classes)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        """
        Performs a forward pass through the ANN_MC_v1 architecture.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = torch.split(input, self.input_size, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in x], 1)
        x = self.act_fn(self.fc_2(x))
        for fc_layer in self.fc_layers:
            x = self.act_fn(fc_layer(x))
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.input_size*self.input_num

class ANN_MC_v2(torch.nn.Module):
    """
    A class representing a multi-layer perceptron (MLP) for multi-class classification.

    Args:
        input_size (int): The size of the input features.
        hidden_sizes (list): A list of integers representing the sizes of the hidden layers.
        act_fn (str): The activation function to be used in the hidden layers.
        output_classes (int): The number of output classes.

    Attributes:
        act_fn (function): The activation function used in the hidden layers.
        fc_layers (torch.nn.ModuleList): A list of fully connected layers.
        fc_n (torch.nn.Linear): The final fully connected layer.
    """

    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_sizes=list, act_fn=str, output_classes=int) -> None:
        super().__init__()
        self.act_fn = _get_act_fn(act_fn)
        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.fc_layers = torch.nn.ModuleList()
        prev = emb_size*input_num
        for size in hidden_sizes:
            self.fc_layers.append(torch.nn.Linear(prev, size))
            prev = size
        self.fc_n = torch.nn.Linear(prev, output_classes)
        self.input_size = input_size
        self.input_num = input_num
    
    def forward(self, input):
        """ Performs forward pass through the network.
        """
        x = torch.split(input, self.input_size, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in x], 1)
        for fc_layer in self.fc_layers:
            x = self.act_fn(fc_layer(x))
        out = self.fc_n(x)
        return out
    
    def get_inputsize(self):
        return self.input_size*self.input_num

class ANN_MO_v1_1(torch.nn.Module):
    """ Artificial Neural Network for Multi Output Regression version 1.1

    In this implementation, all outputs are predicted by the same network and the loss is calculated for all outputs.

    Args:
        input_size (int): The size of the input features.
        hidden_sizes (list): A list of integers representing the hidden dimension of the hidden layers.
        outputs (list): A list of integers representing the outputs. 1 for regression or binary classification and > 1 for multi-class classification.
        act_fn (str): The activation function to be used in the hidden layers.

    Attributes:
        fc_layers (torch.nn.ModuleList): A list of fully connected layers for predicting outputs.
        fc_n (torch.nn.ModuleList): A list of output layers from the hidden layers.
        act_fn (function): The activation function used in the hidden layers.

    """

    def __init__(self, input_size=int, hidden_sizes=list, outputs=list, act_fn=str) -> None:
        super().__init__()

        self.input_size = input_size
        
        # hidden layers for predicting outputs
        self.fc_layers = torch.nn.ModuleList()
        prev = input_size
        for size in hidden_sizes:
            self.fc_layers.append(torch.nn.Linear(prev, size))
            prev = size
        
        # output layers from hidden layers
        self.fc_n = torch.nn.ModuleList()
        self.outputs = outputs
        last_size = hidden_sizes[-1]
        for output in self.outputs:
            self.fc_n.append(torch.nn.Linear(last_size, output))

        self.act_fn = _get_act_fn(act_fn)
        self.input_size = input_size
    
    def forward(self, input):
        x = input

        for layer in self.fc_layers:
            x = self.act_fn(layer(x))

        out = torch.tensor([])
        for i, output in enumerate(self.outputs):
            if output == 1:
                out = torch.cat((out, self.fc_n[i](x)), dim=1)
            else:
                out = torch.cat((out, self.fc_n[i](x)), dim=1)
        
        return out
    
    def get_inputsize(self):
        return self.input_size

class ANN_MO_v1_2(torch.nn.Module):
    """ Artificial Neural Network for Multi Output Regression version 1.2

    This implementation uses separate hidden layers for each output and the loss is calculated for each output.

    Args:
        input_size (int): The size of the input features.
        hidden_first (int): The hidden dimension of the first hidden layer.
        outputs (list): A list of integers representing the outputs. 1 for regression or binary classification and > 1 for multi-class classification.
        hidden_dims (list): A list of integers representing the hidden dimension for each output layer.
        act_fn (str): The activation function to be used in the hidden layers.

    Attributes:
        fc_1 (torch.nn.Linear): The first fully connected layer.
        fc_outputs (torch.nn.ModuleList): A list of fully connected layers for each output.
        act_fn (function): The activation function.

    """

    def __init__(self, input_size=int, hidden_first=int, outputs=list, hidden_dims=list, act_fn=str) -> None:
        super().__init__()

        self.input_size = input_size
        self.outputs = outputs
        self.hidden_dims = hidden_dims

        self.fc_1 = torch.nn.Linear(input_size, hidden_first)
        self.fc_outputs = torch.nn.ModuleList()
        for i, output in enumerate(self.outputs):
            fc_layer_output = torch.nn.ModuleList([
                torch.nn.Linear(hidden_first, self.hidden_dims[i]),
                torch.nn.Linear(self.hidden_dims[i], output)
            ])
            self.fc_outputs.append(fc_layer_output)
        
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        x = self.fc_1(input)

        out = torch.tensor([]).to(input.device)

        for i, output in enumerate(self.outputs):
            x_i = x.clone()
            for layer in self.fc_outputs[i]:
                x_i = layer(self.act_fn(x_i))
            
            if output == 1:
                out = torch.cat((out, x_i), dim=1)
            else:
                out = torch.cat((out, x_i), dim=1)

        return out
    
    def get_inputsize(self):
        return self.input_size

class RNN_regression_v1(torch.nn.Module):

    def __init__(self, input_size=int, emb_size=int, hidden_size=int, act_fn=str, steps=int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.steps = steps
        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
        self.act_fn = _get_act_fn(act_fn)

        self.hidden_tensor = torch.zeros(hidden_size, hidden_size)

        raise NotImplementedError
    
    def forward(self, input_x, hidden_tensor):
        x = self.act_fn(self.fc_1(input_x))
        out, hidden_tensor = self.rnn(x, hidden_tensor)
        out = self.fc(out)
        return out
    
    def get_inputsize(self):
        return self.input_size
    
class RNN_MC_v1(torch.nn.Module):

    def __init__(self, input_size=int, emb_size=int, hidden_size=int, output_classes=int, act_fn=str, steps=int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.steps = steps
        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, output_classes)
        self.act_fn = _get_act_fn(act_fn)

        # hidden tensor initiated to 0
        self.hidden_tensor = torch.zeros(hidden_size, hidden_size)

        raise NotImplementedError
    
    def forward(self, input_x, hidden_tensor):
        x = self.act_fn(self.fc_1(input_x))
        out, hidden_tensor = self.rnn(x, hidden_tensor)
        out = self.fc(out)
        return out
    
    def get_inputsize(self):
        return self.input_size

class LSTM_regression_v0(torch.nn.Module):
    def __init__(self, data_dim=int, hidden_size=int, lstm_layers=int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.data_dim = data_dim
        self.lstm_layers = lstm_layers
        
        self.lstm = torch.nn.LSTM(data_dim, hidden_size, lstm_layers, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, X):
        x = X.view(X.size(0), -1, self.data_dim)
        out_, (hidden, cell) = self.lstm(x)
        out = self.fc_n(out_[:, -1, :])
        return out

class LSTM_MC_v0(torch.nn.Module):
    def __init__(self, data_dim=int, hidden_size=int, lstm_layers=int, output_classes=int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.data_dim = data_dim
        self.lstm_layers = lstm_layers
        self.output_classes = output_classes

        self.lstm = torch.nn.LSTM(data_dim, hidden_size, lstm_layers, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, output_classes)
    
    def forward(self, X):
        x = X.view(X.size(0), -1, self.data_dim)
        out_, (hidden, cell) = self.lstm(x)
        out = self.fc_n(out_[:, -1, :])
        return out
    
class LSTM_regression_v1(torch.nn.Module):

    def __init__(self, data_dim=int, input_num=int, hidden_size=int, num_layers=int, act_fn=str) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_num = input_num
        self.act_fn = _get_act_fn(act_fn)
        self.data_dim = data_dim
        self.combined_size = self.hidden_size*2

        self.fc_1 = torch.nn.Linear(self.data_dim, self.data_dim//2)
        self.lstm = torch.nn.LSTM(self.data_dim//2, self.hidden_size, self.num_layers, batch_first=True)
        self.fc_n = torch.nn.Sequential(
            torch.nn.Linear(self.combined_size, self.combined_size//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.combined_size//2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
    
    def forward(self, X):
        X = X.view(X.size(0), -1, self.data_dim)
        x = self.fc_1(X)
        out_, (hidden, cell) = self.lstm(x)
        out_mean = torch.mean(out_, dim=1)
        combined = torch.cat((out_mean, hidden[-1]), dim=1)
        out = self.fc_n(combined)
        return out
    
    def get_inputsize(self):
        return (self.input_num, self.data_dim)

class LSTM_MC_v1(torch.nn.Module):

    def __init__(self, data_dim=int, input_num=int, hidden_size=int, num_layers=int, output_classes=int, act_fn=str) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_num = input_num
        self.data_dim = data_dim
        self.output_classes = output_classes
        self.act_fn = _get_act_fn(act_fn)
        self.combined_size = self.hidden_size*2

        self.fc_1 = torch.nn.Linear(self.data_dim, self.data_dim//2)
        self.lstm = torch.nn.LSTM(self.data_dim//2, self.hidden_size, self.num_layers, batch_first=True)
        self.fc_n = torch.nn.Sequential(
            torch.nn.Linear(self.combined_size, self.combined_size//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.combined_size//2, self.output_classes)
        )

    def forward(self, X):
        X = X.view(X.size(0), self.input_num, self.data_dim)
        x = self.fc_1(X)
        out_, (hidden, cell) = self.lstm(x)
        out_mean = torch.mean(out_, dim=1)
        combined = torch.cat((out_mean, hidden[-1]), dim=1)
        out = self.fc_n(combined)
        return out
    
    def get_inputsize(self):
        return (self.input_num, self.data_dim)

class ANN_MIMO_v2(torch.nn.Module):
    """ ANN for multiple input and 2 outputs of regression (laptime) and classification (20 positions)
    """

    def __init__(self, input_num=int, input_size=int, hidden_dim=int, emb_dim=int,
                 hidden_output_list=list, act_fn=str) -> None:
        super().__init__()

        self.act_fn = _get_act_fn(act_fn)
        
        self.input_size = input_size * input_num
        self.input_num = input_num
        self.emb_output_dim = emb_dim
        self.hidden_cat_size = self.input_num * self.emb_output_dim
        self.emb_input_dim = input_size

        self.fc_emb = torch.nn.Linear(self.emb_input_dim, self.emb_output_dim)
        self.middle_model = ANN_MO_v1_2(input_size=self.hidden_cat_size, hidden_first=hidden_dim, outputs=[1, 20], hidden_dims=hidden_output_list, act_fn=act_fn)


    def forward(self, input_x):
        inputs_list = torch.split(tensor=input_x, split_size_or_sections=self.emb_input_dim, dim=1)

        hidden_cat = torch.cat([self.act_fn(self.fc_emb(i)) for i in inputs_list], dim=1)
        out = self.middle_model(hidden_cat)

        out_laptime = out[:, :1]
        out_position = out[:, 1:]

        return out_laptime, out_position
    
    def get_inputsize(self):
        return self.input_size

class RNN_reg_v3(torch.nn.Module):
    def __init__(self, data_dim=int, emb_size=int, hidden_size=int, rnn_layers=int):
        super(RNN_reg_v3, self).__init__()
        self.data_dim = data_dim
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.fc_1 = torch.nn.Linear(data_dim, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, rnn_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, X):
        x = X.view(X.size(0), -1, self.data_dim)
        x = self.fc_1(x)
        out_, hidden_ = self.rnn(x)
        out = out_[:, -1, :]
        # out = hidden_.squeeze()
        out = self.fc(out)
        return out
    
    def get_inputsize(self):
        return self.data_dim

class RNN_MC_v3(torch.nn.Module):
    def __init__(self, data_dim=int, emb_size=int, hidden_size=int, num_classes=int, rnn_layers=int):
        super(RNN_MC_v3, self).__init__()
        self.data_dim = data_dim
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnn_layers = rnn_layers
        self.fc_1 = torch.nn.Linear(data_dim, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, rnn_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        x = X.view(X.size(0), -1, self.data_dim)
        x = self.fc_1(x)
        out_, hidden_ = self.rnn(x)
        out = out_[:, -1, :]
        # out = hidden_.squeeze()
        out = self.fc(out)
        return out
    
    def get_inputsize(self):
        return self.data_dim

class LSTM_reg_v3(torch.nn.Module):
    def __init__(self, data_dim=int, emb_size=int, hidden_size=int, lstm_layers=int) -> None:
        super(LSTM_reg_v3, self).__init__()
        self.data_dim = data_dim
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.fc_1 = torch.nn.Linear(data_dim, emb_size)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size, lstm_layers, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, X):
        x = X.view(X.size(0), -1, self.data_dim)
        x = self.fc_1(x)
        out_, (hidden, cell) = self.lstm(x)
        out = self.fc_n(out_[:, -1, :])
        return out

class LSTM_MC_v3(torch.nn.Module):
    def __init__(self, data_dim=int, emb_size=int, hidden_size=int, lstm_layers=int, num_classes=int) -> None:
        super(LSTM_MC_v3, self).__init__()
        self.data_dim = data_dim
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.output_classes = num_classes
        self.fc_1 = torch.nn.Linear(data_dim, emb_size)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size, lstm_layers, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, X):
        x = X.view(X.size(0), -1, self.data_dim)
        x = self.fc_1(x)
        out_, (hidden, cell) = self.lstm(x)
        out = self.fc_n(out_[:, -1, :])
        return out
    
class MyRNN_v2(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes, rnn_layers, fc_layers):
        super(MyRNN_v2, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnn_layers = rnn_layers
        self.fc_layers = fc_layers
        self.emb = torch.nn.Embedding(vocab_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, rnn_layers, batch_first=True)
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(fc_layers)])
        self.fc_n = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, X):
        x = self.emb(X)
        out_, hidden = self.rnn(x)
        out = out_[:, -1, :]
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
        out = self.fc_n(out)
        return out, hidden
    
    def get_inputsize(self):
        return self.vocab_size

class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.fc = nn.Linear(self.hidden_size, self.hidden_size)

  def forward(self, out, hidden):
    hidden_last = hidden[-1]
    hidden_transformed = self.fc(hidden_last)
    attn_scores = torch.bmm(out, hidden_transformed.unsqueeze(2)).squeeze(2)
    attn_weights = torch.softmax(attn_scores, dim=1)
    context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
    return context, attn_weights

class MyRNN_v3(nn.Module):
  def __init__(self, vocab_size, emb_size, hidden_size, num_classes, rnn_layers):
    super(MyRNN_v3, self).__init__()
    self.vocab_size = vocab_size
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.rnn_layers = rnn_layers
    self.combined_size = self.hidden_size * (self.rnn_layers + 2)

    self.fc_emb = nn.Sequential(
        nn.Embedding(self.vocab_size, self.vocab_size//2),
        nn.ReLU(),
        nn.Linear(self.vocab_size//2, 1024),
        nn.ReLU(),
        nn.Linear(1024, self.emb_size)
    )
    self.rnn = nn.RNN(input_size=self.emb_size, hidden_size=self.hidden_size,
                      num_layers=self.rnn_layers, batch_first=True)
    self.attention = Attention(self.hidden_size)

    self.fc_n = nn.Sequential(
        nn.Linear(self.combined_size, self.combined_size//2),
        nn.ReLU(),
        nn.Linear(self.combined_size//2, self.combined_size//4),
        nn.ReLU(),
        torch.nn.Linear(self.combined_size//4, self.combined_size//8),
        nn.ReLU(),
        torch.nn.Linear(self.combined_size//8, 64),
        nn.ReLU(),
        torch.nn.Linear(64, 10),
        nn.ReLU(),
        nn.Linear(10, self.num_classes)
    )

  def forward(self, X):

    x = self.fc_emb(X)
    out_, hidden = self.rnn(x)
    context, attn_weights = self.attention(out_, hidden)
    out_mean = torch.mean(out_, dim=1)
    hidden_flat = hidden.transpose(0,1).reshape(X.size(0), -1)
    combined = torch.cat((context, hidden_flat, out_mean), dim=1)
    out = self.fc_n(combined)
    return out

if __name__ == "__main__":

    # testing model summary to write to file

    # model1 = ANN_MO_v1_2(input_size=10, hidden_first=20, outputs=[1, 20], hidden_dims=[10, 50], act_fn='relu')
    # a = get_model_summary(model1)
    # model2 = ANN_MO_v1_2(input_size=10, hidden_first=20, outputs=[1, 10], hidden_dims=[10, 50], act_fn='sig')
    # a += get_model_summary(model2)
    # print('hello world!')
    # model3 = ANN_MIMO_v2(input_num=20, input_size=500, hidden_dim=200, emb_dim=50, hidden_output_list=[5, 30], act_fn='relu')
    # a += get_model_summary(model3)
    # write_to_file('model_summary_1.txt', a)
    # model_lstm_reg = LSTM_regression_v1(input_size=10, hidden_size=20, num_stacked_layres=2, act_fn='relu')
    # a = get_model_summary(model_lstm_reg)
    # a += '\n NEXT MODEL\n\n'
    # model_lstm_mc = LSTM_MC_v1(input_size=10, hidden_size=20, num_stacked_layres=2, output_classes=20, act_fn='relu')
    # a += get_model_summary(model_lstm_mc)
    # a += '\n NEXT MODEL\n\n'
    # model = LSTM_regression_v1(1, 4, 1)
    # a += get_model_summary(model)
    # write_to_file('delete_this.txt')
    
    # model = MyRNN(4117, 128, 64, 2)
    # a = get_model_summary(model)
    # a += '\n NEXT MODEL\n\n'
    # model = MyRNN_v2(4117, 128, 64, 5, 3, 2)
    # a += get_model_summary(model)
    # write_to_file('delete_this.txt', a)

    model = LSTM_regression_v1(input_size=128, hidden_size=256, num_layers=2, act_fn='relu')
    a = get_model_summary(model)
    a += '\n NEXT MODEL\n\n'
    model = LSTM_MC_v1(input_size=128, hidden_size=256, num_stacked_layres=2, output_classes=20, act_fn='relu')
    a += get_model_summary(model)
    write_to_file('delete_this.txt', a)

    pass
