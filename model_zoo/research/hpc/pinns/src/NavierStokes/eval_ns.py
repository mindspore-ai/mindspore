# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evaluate PINNs for Navier-Stokes equation scenario"""
import numpy as np
from mindspore import context, load_checkpoint, load_param_into_net
from src.NavierStokes.dataset import generate_training_set_navier_stokes
from src.NavierStokes.net import PINNs_navier


def eval_PINNs_navier(ck_path, path, num_neuron=20):
    """
    Evaluation of PINNs for Navier-Stokes equation scenario.

    Args:
        ck_path (str): path of the dataset for Navier-Stokes equation scenario
        path (str): path of the dataset for Navier-Stokes equation
        num_neuron (int): number of neurons for fully connected layer in the network
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    layers = [3, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron,
              num_neuron, 2]

    _, lb, ub = generate_training_set_navier_stokes(10, 10, path, 0)
    n = PINNs_navier(layers, lb, ub)
    param_dict = load_checkpoint(ck_path)
    load_param_into_net(n, param_dict)

    lambda1_pred = n.lambda1.asnumpy()
    lambda2_pred = n.lambda2.asnumpy()
    error_lambda_1 = np.abs(lambda1_pred - 1.0)*100
    error_lambda_2 = np.abs(lambda2_pred - 0.01)/0.01 * 100
    print(f'Error of lambda 1 is {error_lambda_1[0]:.6f}%')
    print(f'Error of lambda 2 is {error_lambda_2[0]:.6f}%')
    return error_lambda_1, error_lambda_2
