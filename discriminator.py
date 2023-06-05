###discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim + num_classes, 1)

    def forward(self, x, c):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc1(output)
        output = F.relu(output)
        output = torch.cat((output, c), dim=1)
        output = self.fc2(output)
        output = torch.sigmoid(output)
        return output.squeeze()
