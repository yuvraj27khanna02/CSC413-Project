import torch
from arch import _get_act_fn

class RNN_regression_v1(torch.nn.Module):
    def __init__(self, input_size=int, emb_size=int, hidden_size=int, act_fn=str) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc_n = torch.nn.Linear(hidden_size, 1)

        self.act_fn = _get_act_fn(act_fn)
        self.hidden_tensor = torch.zeros(num_layers= 1, sequences=1, hidden_size=hidden_size)
    
    def forward(self, input_x, hidden_tensor):
        x = self.act_fn(self.fc_1(input_x))
        out, hidden_tensor = self.rnn(x, hidden_tensor)
        out = self.fc_n(out[:, -1, :])
        return out
    
    def get_inputsize(self):
        return self.input_size

class RNN_regression_v2(torch.nn.Module):
    def __init__(self, input_size=int, input_num=int, emb_size=int, num_layers=int, hidden_sizes=list, act_fn=str) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_num = input_num
        self.hidden_size = hidden_sizes[0]
        self.num_layers = num_layers

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn_layers = torch.nn.ModuleList()
        prev_size = emb_size * input_num  # Update prev_size calculation
        for size in hidden_sizes:
            self.rnn_layers.append(torch.nn.RNN(prev_size, size, batch_first=True))
            prev_size = size

        # self.hidden_tensor = torch.zeros(num_layers, hidden_sizes[0]) 

        self.fc_n = torch.nn.Linear(prev_size, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input_x, hidden_tensor=None):
        if hidden_tensor is None:
            hidden_tensor = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size)
        x = torch.split(input_x, self.input_size, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in x], 1)
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, hidden_tensor = rnn_layer(x, hidden_tensor)
        out = self.fc_n(x[:, -1, :])
        return out, hidden_tensor
    
    def get_inputsize(self):
        return self.input_size * self.input_num
    

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
        self.softmax_ = torch.nn.Softmax(dim=1)
        self.hidden_tensor = torch.zeros(hidden_size, hidden_size)
    
    def forward(self, input_x, hidden_tensor):
        x = self.act_fn(self.fc_1(input_x))
        out, hidden_tensor = self.rnn(x, hidden_tensor)
        out = self.fc(out[:, -1, :])
        return out
    
    def get_inputsize(self):
        return self.input_size


class RNN_MC_v2(torch.nn.Module):

    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_sizes=list, output_classes=int, act_fn=str) -> None:
        super().__init__()

        self.input_size = input_size
        self.input_num = input_num
        self.hidden_size = hidden_sizes[0]

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn_layers = torch.nn.ModuleList()
        prev_size = emb_size * input_num  # Update prev_size calculation
        for size in hidden_sizes:
            self.rnn_layers.append(torch.nn.RNN(prev_size, size, batch_first=True))
            prev_size = size

        self.fc_n = torch.nn.Linear(prev_size, output_classes)  
        self.act_fn = _get_act_fn(act_fn)

    
    def forward(self, input_x, hidden_tensor=None):
        if hidden_tensor is None:
            hidden_tensor = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size)
        x = torch.split(input_x, self.input_size, 1)
        x = torch.cat([self.act_fn(self.fc_1(i)) for i in x], 1)
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, hidden_tensor = rnn_layer(x, hidden_tensor)
        out = self.fc_n(x[:, -1, :])  
        return out, hidden_tensor
    
    def get_inputsize(self):
        return self.input_size * self.input_num
    

