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
"""Export PINNs (Navier-Stokes) model"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import (Tensor, context, export, load_checkpoint,
                       load_param_into_net)
from src.NavierStokes.dataset import generate_training_set_navier_stokes
from src.NavierStokes.net import PINNs_navier


def export_ns(num_neuron, path, ck_file, batch_size, export_format, export_name):
    """
    export PINNs for Navier-Stokes model

    Args:
        num_neuron (int): number of neurons for fully connected layer in the network
        path (str): path of the dataset for Navier-Stokes equation
        ck_file (str): path for checkpoint file
        batch_size (int): batch size
        export_format (str): file format to export
        export_name (str): name of exported file
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    layers = [3, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron,
              num_neuron, 2]

    _, lb, ub = generate_training_set_navier_stokes(10, 10, path, 0)

    n = PINNs_navier(layers, lb, ub)

    param_dict = load_checkpoint(ck_file)
    load_param_into_net(n, param_dict)

    inputs = Tensor(np.ones((batch_size, 3)), mstype.float32)
    export(n, inputs, file_name=export_name, file_format=export_format)
