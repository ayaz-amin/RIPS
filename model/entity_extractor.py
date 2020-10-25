# Custom entity extractor based on convolutional filters

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityExtractor(nn.Module):
    def __init__(self, input_channels, num_objects):
        super(EntityExtractor, self).__init__()

        self.input_channels = input_channels
        self.filters = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(64, num_objects, kernel_size=1),
                nn.Sigmoid()
                )

    def obs_to_torch(self, obs):
        height, width = obs.shape[0], obs.shape[1]
        obs = torch.from_numpy(obs.copy()).float()
        return obs.view(1, self.input_channels, height, width)

    def parsed_objects(self, z):
        object_blackboard = torch.zeros(z.shape[2], z.shape[3])
        
        z = z.view(z.shape[1], z.shape[2], z.shape[3])
        for object_idx in range(z.shape[0]):
            for r in range(z.shape[1]):
                for c in range(z.shape[2]):
                    if z[object_idx, r, c] != 0:
                        object_blackboard[r][c] = object_idx

        return object_blackboard.detach().numpy()

    def forward(self, obs):
        obs = self.obs_to_torch(obs)
        z = self.filters(obs)
        return self.parsed_objects(z)
