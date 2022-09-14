import torch.nn as nn
import torch.nn.functional as F
import torch


class RecommendFailed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpu = config.use_gpu
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.drop = nn.Dropout(config.dp)

        if self.gpu:
            self.fc1 = self.fc1.cuda()

    def forward(self, x):
        x = self.fc1(x)

        return x
