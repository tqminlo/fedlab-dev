import torch
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report

num_clients = 100
num_classes = 10
seed = 2021
hist_color = '#4169E1'


trainset = torchvision.datasets.CIFAR10(root="../../../../data/CIFAR10/",
                                        train=True, download=True)


hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                     num_clients,
                                     balance=None,
                                     partition="dirichlet",
                                     dir_alpha=0.3,
                                     seed=seed)

# print(hetero_dir_part.client_dict)

for key in hetero_dir_part.client_dict.keys():
    print(key, len(hetero_dir_part.client_dict[key]))

