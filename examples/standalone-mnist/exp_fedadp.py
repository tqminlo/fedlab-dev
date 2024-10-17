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
from fedlab.contrib.algorithm.fedadp import FedAdpServerHandler, FedAdpSerialClientTrainer
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.pathfedadp_mnist import PathFedAdpMNIST

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_clients", type=int, default=100)
parser.add_argument("--com_round", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--num_iid", type=int)
parser.add_argument("--x_class", type=int)
parser.add_argument("--exp", type=str)
parser.add_argument("--alg", type=str, default="adp")

args = parser.parse_args()

model = CNN_MNIST()

if args.alg == "adp":
    # server
    handler = FedAdpServerHandler(model, args.com_round, args.total_clients)    # sample_ratio = 1
    # client
    trainer = FedAdpSerialClientTrainer(model, args.total_clients, cuda=False)
else:   # alg == "avg"
    handler = FedAvgServerHandler(model, args.com_round, args.total_clients)
    trainer = FedAvgSerialClientTrainer(model, args.total_clients, cuda=False)


dataset = PathFedAdpMNIST(
    root="../../datasets/mnist/",
    path=f"../../datasets/mnist/{args.exp}",
    num_clients=args.total_clients,
    num_iid=args.num_iid,
    x_class=args.x_class,
)
dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

handler.setup_dataset(dataset)
# main
pipeline = StandalonePipeline(handler, trainer)
pipeline.main()
