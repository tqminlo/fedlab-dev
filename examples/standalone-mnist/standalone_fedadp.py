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
from fedlab.models.cnn import CNN_MNIST
from fedlab.contrib.algorithm.fedadp import FedAdpServerHandler
from fedlab.contrib.algorithm.fedadp import FedAdpSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_clients", type=int, default=100)
parser.add_argument("--com_round", type=int)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float)

args = parser.parse_args()

# model = MLP(784, 10)
model = CNN_MNIST()

# server
handler = FedAdpServerHandler(
    model, args.com_round, args.total_clients, args.sample_ratio
)

# client
trainer = FedAdpSerialClientTrainer(model, args.total_clients, cuda=False)
dataset = PathologicalMNIST(
    root="../../datasets/mnist/",
    path="../../datasets/mnist/",
    num_clients=args.total_clients,
)
dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

handler.setup_dataset(dataset)
# main
pipeline = StandalonePipeline(handler, trainer)
pipeline.main()
