import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.num_classes = 8
        self.num_layers = 1
        self.input_size = 13
        self.hidden_size = 100

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True)
        self.fc_1 = nn.Linear(self.hidden_size, 128)
        self.fc_2 = nn.Linear(128, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (out, __) = self.lstm(x, (h_0, c_0))
        out = out.view(-1, self.hidden_size)
        out = self.relu(out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out