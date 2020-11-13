from copy import copy
import numpy as np

import torch
import torch.distributions as dist

def squeeze_params(model):
    squeezed_params = []
    for param in model.parameters():
        if param.requires_grad:
            squeezed_params.append(param.data)

    return squeezed_params

def sample_models(model, num_pop):
    model_zoo = []
    for _ in range(num_pop):
        child = copy(model)
        child_params = squeeze_params(model)
        for param in child_params:
            if param.requires_grad:
                mean = param.data
                stddev = torch.ones_like(mean)
                p = dist.Normal(mean, stddev).sample()
                param.data = p
        model_zoo.append(child)
    return model_zoo
