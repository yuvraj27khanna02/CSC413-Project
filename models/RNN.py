import torch
from arch import _get_act_fn
import torch.nn.functional as F


class RNN_regression_v1(torch.nn.Module):
    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_size=int, num_layers=int, act_fn=str, dropout_rate=0.5) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_num = input_num
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc_n = torch.nn.Linear(hidden_size*2, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input_x):
        x = input_x.view(-1, self.input_num, self.input_size)
        hidden = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        x = self.fc_1(x)
        x = self.act_fn(x)
        out, hn = self.rnn(x, hidden)
        out = self.dropout(out)
        out_mean = torch.mean(out, dim=1)
        out = self.fc_n(out_mean)
        return out, hn

    def get_inputsize(self):
        return self.input_size

class RNN_regression_v2(torch.nn.Module):
    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_size=int, num_layers=int, act_fn=str, dropout_rate=0.5) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_num = input_num
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn_layers = torch.nn.ModuleList([
            torch.nn.RNN(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True) for _ in range(num_layers)
        ])
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc_n = torch.nn.Linear(hidden_size*2, 1)
        self.act_fn = _get_act_fn(act_fn)
    
    def forward(self, input_x):
        x = input_x.view(-1, self.input_num, self.input_size)
        hidden = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        x = self.fc_1(x)
        x = self.act_fn(x)
        for rnn_layer in self.rnn_layers:
            out, hn = rnn_layer(x, hidden)
            x = out
        out = self.dropout(out)
        out_mean = torch.mean(out, dim=1)
        out = self.fc_n(out_mean)
        return out, hn
    
    def get_inputsize(self):
        return self.input_size  

class RNN_MC_v1(torch.nn.Module):

    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_size=int, num_layers=int, output_classes=int, act_fn=str, dropout_rate=0.5) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_num = input_num
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc_n = torch.nn.Linear(hidden_size*2, output_classes)
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input_x):
        x = input_x.view(-1, self.input_num, self.input_size)
        hidden = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        x = self.fc_1(x)
        x = self.act_fn(x)
        out, hn = self.rnn(x, hidden)
        out = self.dropout(out)
        out_mean = torch.mean(out, dim=1)  
        out = self.fc_n(out_mean)
        out = self.softmax_(out)
        return out, hn
    
    def get_inputsize(self):
        return self.input_size

class RNN_MC_v2(torch.nn.Module):

    def __init__(self, input_size=int, input_num=int, emb_size=int, hidden_size=int, num_layers=int, output_classes=int, act_fn=str, dropout_rate=0.5) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_num = input_num
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.fc_1 = torch.nn.Linear(input_size, emb_size)
        self.rnn_layers = torch.nn.ModuleList([
            torch.nn.RNN(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True) for _ in range(num_layers)
        ])
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc_n = torch.nn.Linear(hidden_size*2, output_classes)
        self.act_fn = _get_act_fn(act_fn)
        self.softmax_ = torch.nn.Softmax(dim=1)
    
    def forward(self, input_x):
        x = input_x.view(-1, self.input_num, self.input_size)
        hidden = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        x = self.fc_1(x)
        x = self.act_fn(x)
        for rnn_layer in self.rnn_layers:
            out, hn = rnn_layer(x, hidden)
            x = out
        out = self.dropout(out)
        out_mean = torch.mean(out, dim=1)  
        out = self.fc_n(out_mean)
        out = self.softmax_(out)
        return out, hn
    
    def get_inputsize(self):
        return self.input_size