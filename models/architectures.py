import torch
import torchsummary

def _get_act_fn(act_fn=str):
    if act_fn == 'relu':
        return torch.nn.ReLU()
    if act_fn == 'sig':
        return torch.nn.Sigmoid()
    if act_fn == 'tanh':
        return torch.nn.Tanh()

    
class MLP_BC_v1(torch.nn.Module):
    """ Multilayer Perceptron for Binary Classification version 1
    """

    def __init__(self, input_size=int, hidden_size=int, num_layers=int,  act_fn=str) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        x = self.act_fn(self.fc_1(input))
        out = self.fc_n(x)
        return out

class MLP_regression_v1(torch.nn.Module):
    """ MultiLayer Perceptron for Regression version 1
    """

    def __init__(self, input_size=int, hidden_size=int, num_layers=int, act_fn=str) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        x = self.act_fn(self.fc_1(input))
        out = self.fc_n(x)
        return out

class MLP_MC_v1(torch.nn.Module):
    """ Multilayer Perceptron for Multiclass Classification version 1
    """

    def __init__(self, input_size=int, hidden_size=int, output_classes=int, num_layers=int,  act_fn=str) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size, output_classes)
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input):
        x = self.act_fn(self.fc_1(input))
        out = self.softmax_(self.fc_n(x))
        return out

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

    def __init__(self, input_num=int, hidden_size=int, act_fn=str) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(input_num, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input):
        """Forward pass of the neural network.
        
        Args:
            input (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.act_fn(self.fc_1(input))
        out = self.fc_n(x)
        return out

class ANN_regression_v2(torch.nn.Module):
    """MultiLayer Perceptron for Regression version 2 with tapering hidden dimensions

    Args:
            input_size (int): The size of the input features.
            hidden_sizes (List[int]): A list of hidden layer sizes. The number of hidden layers will be determined by the length of this list.
            act_fn (str): The activation function to use in the hidden layers.
    """

    def __init__(self, input_size=int, hidden_sizes=list, act_fn=str) -> None:
        super().__init__()
        self.act_fn = _get_act_fn(act_fn)
        self.fc_layers = torch.nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.fc_layers.append(torch.nn.Linear(prev_size, size))
            prev_size = size
        self.fc_n = torch.nn.Linear(prev_size, 1)
    
    def forward(self, input):
        """
        Performs forward pass through the ANN_regression_v2 model.
        
        Args:
            input (torch.Tensor): The input tensor.
        """
        x = input
        for fc_layer in self.fc_layers:
            x = self.act_fn(fc_layer(x))
        out = self.fc_n(x)
        return out

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

    def __init__(self, input_size=int, hidden_size=int, act_fn=str, output_classes=int) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_n = torch.nn.Linear(hidden_size, output_classes)
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input):
        """
        Performs a forward pass through the ANN_MC_v1 architecture.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.act_fn(self.fc_1(input))
        out = self.softmax_(self.fc_n(x))
        return out

class ANN_MC_v2(torch.nn.Module):
    """
    A class representing a multi-layer perceptron (MLP) with softmax activation for multi-class classification.

    Args:
        input_size (int): The size of the input features.
        hidden_sizes (list): A list of integers representing the sizes of the hidden layers.
        act_fn (str): The activation function to be used in the hidden layers.
        output_classes (int): The number of output classes.

    Attributes:
        act_fn (function): The activation function used in the hidden layers.
        softmax_ (torch.nn.Softmax): The softmax activation function used for the final output.
        fc_layers (torch.nn.ModuleList): A list of fully connected layers.
        fc_n (torch.nn.Linear): The final fully connected layer.
    """

    def __init__(self, input_size, hidden_sizes=list, act_fn=str, output_classes=int) -> None:
        super().__init__()
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
        self.fc_layers = torch.nn.ModuleList()
        prev = input_size
        for size in hidden_sizes:
            self.fc_layers.append(torch.nn.Linear(prev, size))
            prev = size
        self.fc_n = torch.nn.Linear(prev, output_classes)
    
    def forward(self, input):
        """ Performs forward pass through the network.
        """
        x = input
        for fc_layer in self.fc_layers:
            x = self.act_fn(fc_layer(x))
        out = self.softmax_(self.fc_n(x))
        return out

class ANN_MO_v1(torch.nn.Module):
    """ Artificial Neural Network for Multi Output Regression version 1.

    In this implementation, 
    """

    def __init__(self, ) -> None:
        super().__init__()

class RNN_MC_v1(torch.nn.Module):
    """ Recurrent Neural Network for Multiclass Classification version 1
    """

    raise NotImplementedError

    def __init__(self, input_size=int, hidden_size=int, output_classes=int, num_layers=int,  act_fn=str) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_input_to_hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.fc_input_to_output = torch.nn.Linear(input_size + hidden_size, output_classes)
        self.softmax_ = torch.nn.Softmax(dim=1)

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, output_classes)
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input, hidden):

        x = torch.cat((input, hidden), dim=1)
        hidden = self.fc_input_to_hidden(x)
        output = self.fc_input_to_output(x)
        output = self.softmax_(output)
        return output, hidden


