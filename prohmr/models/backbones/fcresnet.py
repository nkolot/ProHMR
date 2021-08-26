import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=True):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(nn.Dropout(0.5))
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)

class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=True):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5))

    def forward(self, x):
        return x + self.fc_block(x)

class FCResNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=True):
        super(FCResNet, self).__init__()
        module_list = [FCBlock(in_channels, hidden_channels, batchnorm=batchnorm, activation=activation, dropout=dropout)]
        for i in range(num_layers):
            module_list.append(FCResBlock(hidden_channels, hidden_channels, batchnorm=batchnorm, activation=activation, dropout=dropout))
        if hidden_channels != out_channels:
            module_list.append(FCBlock(hidden_channels, out_channels, batchnorm=batchnorm, activation=activation, dropout=dropout))
        self.fc_layers = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_layers(x)

def fcresnet(cfg):
    return FCResNet(cfg.MODEL.BACKBONE.IN_CHANNELS, cfg.MODEL.BACKBONE.HIDDEN_CHANNELS,
                    cfg.MODEL.BACKBONE.OUT_CHANNELS, cfg.MODEL.BACKBONE.NUM_LAYERS)
