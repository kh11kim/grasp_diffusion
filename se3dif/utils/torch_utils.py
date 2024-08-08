import os
import torch
import collections
import numpy as np
import json

specifications_filename = "params.json"

def load_experiment_specifications(experiment_param_path):

    #filename = os.path.join(experiment_directory, specifications_filename)

    #if not os.path.isfile(filename):
    if not os.path.isfile(experiment_param_path):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"params.json"'.format(experiment_param_path)
        )

    return json.load(open(experiment_param_path))


def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def to_numpy(x):
    return x.detach().cpu().numpy()


def to_torch(x, device='cpu'):
    if isinstance(x, list):
        return torch.Tensor(x).float().to(device)
    else:
        return torch.from_numpy(x).float().to(device)
