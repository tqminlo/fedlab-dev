from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch

sys.path.append("../../")
torch.manual_seed(0)

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_CIFAR10
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_clients", type=int, default=100)
parser.add_argument("--balance", default=True)
parser.add_argument("--unbalance_sgm", default=0)
parser.add_argument("--dir_alpha", default=100)

parser.add_argument("--com_round", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float)

args = parser.parse_args()

model = CNN_CIFAR10()

# server
handler = FedAvgServerHandler(
    model, args.com_round, args.total_clients, args.sample_ratio
)

# client
trainer = FedAvgSerialClientTrainer(model, args.total_clients, cuda=False)
dataset = PartitionedCIFAR10(
    root="../../datasets/cifar10/",
    path="../../datasets/cifar10/",
    dataname="cifar10",
    num_clients=args.total_clients,
    balance=args.balance,
    partition="dirichlet",
    unbalance_sgm=args.unbalance_sgm,
    dir_alpha=args.dir_alpha,
)
dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

handler.setup_dataset(dataset)
# main
pipeline = StandalonePipeline(handler, trainer)
pipeline.main()
