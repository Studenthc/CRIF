import torch.nn as nn
import torch.nn.functional as F
import torch


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpu = config.use_gpu
        self.input_dim = config.input_dim + 8
        self.hidden_dim = config.hidden_dim
        self.output_dim = 1
        self.pre1 = nn.Linear(1, 8)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(config.dp)

        if self.gpu:
            self.pre1 = self.pre1.cuda()
            self.fc1 = self.fc1.cuda()
            self.fc3 = self.fc3.cuda()
            self.drop = self.drop.cuda()

    def forward(self, x):
        a_pre = self.pre1(x[-1].unsqueeze(-1))
        h1 = self.fc1(torch.cat((x[:-1], a_pre)))
        h1 = F.relu(h1)
        h1 = self.drop(h1)
        h3 = self.fc3(h1)
        h3 = self.tanh(h3)
        h3 = h3 * 0.1
        h3 = h3.squeeze(-1)

        return h3
