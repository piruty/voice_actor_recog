import torch
from torch import nn
import torch.nn.functional as F


activation = torch.nn.Tanh()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(20, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self):
        torch.save(self.state_dict(), './files/model.pth')
