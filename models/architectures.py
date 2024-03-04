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
        softmax_ (torch.nn.Softmax): The softmax function used for outputs with more than one class.

    """

    def __init__(self, input_size=int, hidden_sizes=list, outputs=list, act_fn=str) -> None:
        super().__init__()
        
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
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input):
        x = input

        for layer in self.fc_layers:
            x = self.act_fn(layer(x))

        out = torch.tensor([])
        for i, output in enumerate(self.outputs):
            if output == 1:
                out = torch.cat((out, self.fc_n[i](x)), dim=1)
            else:
                out = torch.cat((out, self.softmax_(self.fc_n[i](x))), dim=1)
        
        return out

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
        softmax_ (torch.nn.Softmax): The softmax function.

    """

    def __init__(self, input_size=int, hidden_first=int, outputs=list, hidden_dims=list, act_fn=str,) -> None:
        super().__init__()

        self.outputs = outputs
        self.hidden_dims = hidden_dims

        self.fc_1 = torch.nn.Linear(input_size, hidden_first)
        self.fc_outputs = torch.nn.ModuleList()
        for i, output in enumerate(self.outputs):
            fc_layer_output = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_first, self.hidden_dims[i])],
                [torch.nn.Linear(self.hidden_dims[i], output)]
            )
            self.fc_outputs.append(fc_layer_output)
        
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input):
        x = self.act_fn(self.fc_1(input))

        out = torch.tensor([])

        for i, output in enumerate(self.outputs):
            for layer in self.fc_outputs[i]:
                x = self.act_fn(layer(x))
            
            if output == 1:
                out = torch.cat((out, layer(x)), dim=1)
            else:
                out = torch.cat((out, self.softmax_(layer(x))), dim=1)

        return out



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

