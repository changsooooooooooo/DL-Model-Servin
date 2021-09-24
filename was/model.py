import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.first_layer_list = nn.ModuleList(
            [nn.Conv2d(1, 16, (3, 1), padding=0), nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d((2, 1))]
        )
        self.second_layer_list = nn.ModuleList(
            [nn.Conv2d(16, 16 * 2, (3, 1), padding=0), nn.BatchNorm2d(16 * 2), nn.ELU(), nn.MaxPool2d((2, 1))]
        )
        self.third_layer_list = nn.ModuleList(
            [nn.Conv2d(16 * 2, 16 * 4, (3, 1), padding=0), nn.BatchNorm2d(16 * 4), nn.ELU(), nn.MaxPool2d((2, 1))]
        )
        self.fourth_layer_list = nn.ModuleList(
            [nn.Conv2d(16 * 4, 16 * 8, (3, 1), padding=0), nn.BatchNorm2d(16 * 8), nn.ELU(), nn.MaxPool2d((2, 1))]
        )
        self.fifth_layer_list = nn.ModuleList(
            [nn.Conv2d(16 * 8, 16 * 16, (3, 1), padding=0), nn.BatchNorm2d(16 * 16), nn.ELU(), nn.MaxPool2d((2, 1))]
        )
        self.sixth_layer_list = nn.ModuleList(
            [nn.Conv2d(16 * 16, 16 * 32, (3, 1), padding=0), nn.BatchNorm2d(16 * 32), nn.ELU(), nn.MaxPool2d((2, 1))]
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(46080, 512), nn.Linear(512, 256), nn.Linear(256, 32), nn.Linear(32, 16), nn.Linear(16, 1)]
        )

    def forward(self, x):
        for layer in self.first_layer_list:
            x = layer(x)
        for layer in self.second_layer_list:
            x = layer(x)
        for layer in self.third_layer_list:
            x = layer(x)
        for layer in self.fourth_layer_list:
            x = layer(x)
        for layer in self.fifth_layer_list:
            x = layer(x)
        for layer in self.sixth_layer_list:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x
