# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
import numpy as np
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)

        return serialized_parameters

    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters

    @staticmethod
    def fedadp_aggregate(init_model, serialized_params_list, t, weights=None, pre_thetas=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        weights = weights / torch.sum(weights)
        # print("----1---- weights(ratio num data):", weights)

        '''Calculate weights from init-model, client-models, weights(num-data)'''
        gradients_list = [(init_model-w)/0.02 for w in serialized_params_list]
        gradients_list = [t.numpy() for t in gradients_list]
        gradients_list = torch.tensor(gradients_list)
        weights_2d = weights.unsqueeze(1)
        gradient_global = gradients_list * weights_2d
        gradient_global = torch.sum(gradient_global, 0)
        cosin_similary_list = [torch.nn.CosineSimilarity(dim=0)(gradient_global, grad_k) for grad_k in gradients_list]
        thetas = torch.tensor([torch.arccos(cos).item() for cos in cosin_similary_list])
        # print("----1---- thetas:", thetas)
        # print("----2---- pre_thetas:", pre_thetas)

        thetas_smooth = thetas
        # thetas_smooth = pre_thetas * (t-1)/t + thetas * 1/t

        def gompertz_func(theta, alpha=5.):
            return alpha * (1 - np.exp(-np.exp(-alpha*(theta-1))))
        # print("----3---- thetas_smooth:", thetas_smooth)
        re_weights = gompertz_func(thetas_smooth, alpha=5.)
        # print("----4---- re_weights:", re_weights)

        re_weights = re_weights / torch.sum(re_weights)
        # print("----5---- re_weights:", re_weights)

        assert torch.all(re_weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * re_weights, dim=-1)

        return serialized_parameters, thetas_smooth
