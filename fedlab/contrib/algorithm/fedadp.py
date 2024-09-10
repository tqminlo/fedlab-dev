from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils.aggregator import Aggregators
from ...utils.serialization import SerializationTool

##################
#
#      Server
#
##################


class FedAdpServerHandler(SyncServerHandler):
    """FedAdp server handler."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        num_data = [ele[1] for ele in buffer]
        init_model = self.model_parameters
        # print("----5----:", self.round)
        serialized_parameters, self.thetas = Aggregators.fedadp_aggregate(init_model, parameters_list, self.round,
                                                                          num_data, self.thetas)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


##################
#
#      Client
#
##################


class FedAdpClientTrainer(SGDClientTrainer):
    """Federated client with local SGD solver."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        num_data = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedadp_aggregate(self._model, parameters_list, num_data)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedAdpSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        data_size = 0
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters, data_size]
