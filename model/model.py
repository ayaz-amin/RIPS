# Methods taken from https://github.com/tomsilver/policies_logic_programs/blob/master/dsl.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .entity_extractor import EntityExtractor


def out_of_bounds(r, c, shape):
    return (r < 0 or c < 0 or r >= shape[0] or c >= shape[1])

def shifted(direction, local_program, cell, obs):
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
    return local_program(new_cell, obs)

def cell_is_value(value, cell, obs):
    if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
        focus = None
    else:
        focus = obs[cell[0], cell[1]]

    return (focus == value)

def at_cell_with_value(value, local_program, obs):
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
    else:
        cell = matches[0]
    return local_program(cell, obs)

def scanning(direction, true_condition, false_condition, cell, obs, max_timeout=50):
    if cell is None:
        return False

    for _ in range(max_timeout):
        cell = (cell[0] + direction[0], cell[1] + direction[1])

        if true_condition(cell, obs):
            return True

        if false_condition(cell, obs):
            return False

        # prevent infinite loops
        if out_of_bounds(cell[0], cell[1], obs.shape):
            return False

    return False

# My classes
class Model(nn.Module):
    # Container for program synthesis model
    def __init__(self, input_channels, object_types, action_types, num_programs):
        super(Model, self).__init__()

        self.feature_extractor = EntityExtractor(input_channels, object_types)
        self.action_types = action_types
        self.programs = nn.ModuleList()
        for i in range(num_programs):
            self.programs.append(AtActionCell(object_types, action_types))

    def forward(self, obs):
        obs = self.feature_extractor(obs)
        action_probs = torch.zeros(self.action_types)
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                for program in self.programs:
                    condition, action = program((r, c), obs)
                    if condition:
                        action_probs[action] += 1

        normalized_action_probs = F.softmax(action_probs, dim=0)
        return dist.Categorical(normalized_action_probs) 


class AtActionCell(nn.Module):
    def __init__(self, object_types, action_types):
        super(AtActionCell, self).__init__()
        self.object_types = nn.Parameter(torch.ones(object_types))
        self.positive_object_types = nn.Parameter(torch.ones(object_types))
        self.negative_object_types = nn.Parameter(torch.ones(object_types))

        self.action_types = nn.Parameter(torch.ones(action_types))
        self.direction_types = nn.Parameter(torch.ones(8))
        self.directions = [
            (1, 0), (0, 1),
            (-1, 0), (0, -1),
            (1, 1), (-1, 1),
            (1, -1), (-1, -1)
        ]

        self.is_not = nn.Parameter(torch.ones(2))

    def forward(self, cell, obs):
        # Sample function parameters
        object_probs = F.softmax(self.object_types, dim=0)
        positive_object_probs = F.softmax(self.positive_object_types, dim=0)
        negative_object_probs = F.softmax(self.negative_object_types, dim=0)

        action_probs = F.softmax(self.action_types, dim=0)
        direction_probs = F.softmax(self.direction_types, dim=0)

        sample_object = dist.Categorical(object_probs).sample()
        sample_positive_object = dist.Categorical(positive_object_probs).sample()
        sample_negative_object = dist.Categorical(negative_object_probs).sample()

        sample_action = dist.Categorical(action_probs).sample()
        sample_direction = dist.Categorical(direction_probs).sample()

        direction = self.directions[sample_direction]

        is_not_probs = F.softmax(self.is_not, dim=0)
        is_not = dist.Categorical(is_not_probs).sample()

        # Main program
        condition = at_cell_with_value(
            sample_object, 
            lambda cell, obs : scanning(
                direction,
                lambda cell, obs : cell_is_value(sample_positive_object, cell, obs),
                lambda cell, obs : cell_is_value(sample_negative_object, cell, obs),
                cell,
                obs
            ),
            obs
        )

        if is_not.item():
            condition = not condition

        return condition, sample_action
