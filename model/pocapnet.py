import torch as t 

from torch import nn
from utils import hparams
from model.audio_branch import audioBranch
from model.mstcn import MultiStageModel


class AvgResBlock(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_input, padding = 1, kernel_size = 3)

        self.conv2 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_input, padding = 1, kernel_size = 3)

        self.conv3 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_input, padding = 1, kernel_size = 3)

        self.conv4 = nn.Conv1d(in_channels = n_input, 
            out_channels = n_input, padding = 1, kernel_size = 3)

        self.bn1 = nn.BatchNorm1d(n_input)
        self.bn2 = nn.BatchNorm1d(n_input)
        self.bn3 = nn.BatchNorm1d(n_input)
        self.bn4 = nn.BatchNorm1d(n_input)

        self.relu = nn.ReLU()

        self.avg_pool = nn.AvgPool1d(n_input)


    def forward(self, x):
        res = x

        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x += res
        x = self.relu(x)
        res = x

        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.bn4(self.conv4(x))
        x += res
        x = self.relu(x)

        x = t.permute(x, (0, 2, 1))
        x = self.avg_pool(x)
        return t.permute(x, (0, 2, 1))



class PoCaPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.vb = AvgResBlock(n_input = hparams.resolution)
        
        self.ab1 = audioBranch(n_channel = 349)
        self.ab2 = audioBranch(n_channel = 349)
        self.ab3 = audioBranch(n_channel = 349)
        self.ab = AvgResBlock(n_input = 3)

        self.project = nn.Linear(in_features = 2048, out_features = 1024)

        self.mstcn = MultiStageModel(hparams.num_stages,
            hparams.num_layers, hparams.num_f_maps, hparams.dim, 
            hparams.num_classes, hparams.causal_conv)
            
        self.logsoftmax = t.nn.LogSoftmax(dim = 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x, c1, c2, c3, pe, h):
        x = self.vb(x) #(B, 1, 1024)

        c1 = self.ab1(c1) #(B, 1, 1024)
        c2 = self.ab2(c2)
        c3 = self.ab3(c3)
        c = t.cat((c1, c2, c3), dim = 1) #(B, 3, 1024)
        c = self.ab(c) #(B, 1, 1024)

        h = h.expand(x.shape[0], -1, -1)
        
        f = t.cat((x,c), dim = 2) #(B, 1, 2048)
        f = self.relu(self.project(f)) #(B, 1, 1024)
        #f += pe
        f += h 
        f = t.permute(f, (1, 2, 0)) #(1, 1024, B)
        
        
        m = self.mstcn(f) #(n_stage, 1, n_class, B)
        m = m.squeeze(dim = 1) #(n_stage, n_class, B)
        m = t.permute(m, (2,1,0)) #(B, n_class, n_stage)
        
        return m #(batch_size, num_classes, num_stages)