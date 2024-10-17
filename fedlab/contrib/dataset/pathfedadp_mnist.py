
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.utils import shuffle
from .basic_dataset import FedDataset, BaseDataset
from ...utils.dataset.functional import noniid_slicing, random_slicing


class PathFedAdpMNIST(FedDataset):
    """The partition stratigy in FedAvg. See http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com

    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        shards (int, optional): Sort the dataset by the label, and uniformly partition them into shards. Then
        download (bool, optional): Download. Defaults to True.
    """

    def __init__(self, root, path, num_clients=10, num_iid=5, x_class=2) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.num_iid = num_iid
        self.x_class = x_class

    def preprocess(self, download=True):
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)

        if os.path.exists(os.path.join(self.path, "train")) is not True:
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        # train
        mnist = torchvision.datasets.MNIST(self.root, train=True, download=self.download,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))])
                                           )

        ### Sửa ở đây nha
        X_train = []
        Y_train = []
        for x, y in mnist:
            X_train.append(x)
            Y_train.append(y)
        X_train, Y_train = shuffle(X_train, Y_train)
        # X_train, Y_train = np.array(X_train), np.array(Y_train).astype(np.int64)
        # check_label = [0] * 10
        # for l in Y_train:
        #     check_label[l] += 1
        # print("all: ", check_label)
        # idx_order = np.argsort(Y_train)[::1]
        # Y_train = Y_train[idx_order]
        # X_train = X_train[idx_order]

        for i in range(self.num_iid):
            X_train_node = X_train[600 * i: 600 * (i+1)]
            Y_train_node = Y_train[600 * i: 600 * (i+1)]
            check_label = [0] * 10
            for l in Y_train_node:
                check_label[l] += 1
            print(i, check_label)
            dataset = BaseDataset(X_train_node, Y_train_node)
            torch.save(dataset, os.path.join(self.path, "train", f"data{i}.pkl"))

        start_id = 600 * self.num_iid
        for i in range(self.num_iid, self.num_clients):
            all_id = shuffle(np.arange(10))
            client_class = all_id[:self.x_class]
            # num_data_per_niid = 600 // self.x_class
            X_train_node, Y_train_node = [], []
            num_data_client = 0
            for j in range(start_id, 60000):
                if Y_train[j] in client_class:
                    X_train_node.append(X_train[j])
                    Y_train_node.append(Y_train[j])
                    num_data_client += 1
                if num_data_client == 600:
                    start_id = j
                    break

            check_label = [0] * 10
            for l in Y_train_node:
                check_label[l] += 1
            print(i, check_label)
            dataset = BaseDataset(X_train_node, Y_train_node)
            torch.save(dataset, os.path.join(self.path, "train", f"data{i}.pkl"))
        ### Nguyên khối cần sửa đây nha

        # test
        mnist_test = torchvision.datasets.MNIST(
            self.root,
            train=False,
            download=self.download,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]),
        )
        test_samples, test_labels = [], []
        for x, y in mnist_test:
            test_samples.append(x)
            test_labels.append(y)
        test_dataset = BaseDataset(test_samples, test_labels)
        torch.save(test_dataset, os.path.join(self.path, "test", "test.pkl"))

    def get_dataset(self, id=None, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
            cid (int): client id
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        if type == "train":
            dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        else:
            dataset = torch.load(os.path.join(self.path, "test", "test.pkl"))
        return dataset

    def get_dataloader(self, id=None, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
