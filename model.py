import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.modules):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        _, (out, __) = self.lstm(x, (h_0, c_0))
        out = out.view(-1, self.hidden_size)
        out = self.relu(out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out