import copy
import torch as t
import torch.nn as nn

from utils import hparams


class ResBlock(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.conv1 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_output, padding = 1, kernel_size = 3)
        
        self.conv2 = nn.Conv1d(in_channels = n_output, 
            out_channels = n_output, padding = 1, kernel_size = 3)

        self.downsample = nn.Conv1d(in_channels = n_input, 
            out_channels = n_output, kernel_size = 1)

        self.bn1 = nn.BatchNorm1d(n_output)
        self.bn2 = nn.BatchNorm1d(n_output)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.n_input != self.n_output:
            res = self.downsample(res)

        x += res
        return self.relu(x)


class audioBranch(nn.Module):
    def __init__(self, n_channel, n_feat = 128):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels = n_channel, 
            out_channels = n_feat, kernel_size = 1)
        
        self.bn1 = nn.BatchNorm1d(n_feat, momentum = 0.01)

        self.relu = nn.ReLU()

        self.layer1 = ResBlock(n_input = n_feat, n_output = n_feat // 2)
        self.layer2 = ResBlock(n_input = n_feat // 2, n_output = n_feat // 4)
        self.layer3 = ResBlock(n_input = n_feat // 4, n_output = n_feat // 8)
        self.layer4 = ResBlock(n_input = n_feat // 8, n_output = n_feat // 16)

        self.aud_pool = nn.AvgPool1d(kernel_size = n_feat // 16)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = t.permute(x, (0, 2, 1))
        x = self.aud_pool(x)
        x = t.permute(x, (0, 2, 1))

        return x