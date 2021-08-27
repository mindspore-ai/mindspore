# Copyright 2020 Huawei Technologies Co., Ltd
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
"""test gat model."""
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
from gat import GAT

context.set_context(mode=context.GRAPH_MODE)


def test_GAT():
    ft_sizes = 1433
    num_class = 7
    num_nodes = 2708
    hid_units = [8]
    n_heads = [8, 1]
    activation = nn.ELU()
    residual = False
    input_data = Tensor(
        np.array(np.random.rand(1, 2708, 1433), dtype=np.float32))
    biases = Tensor(np.array(np.random.rand(1, 2708, 2708), dtype=np.float32))
    net = GAT(ft_sizes,
              num_class,
              num_nodes,
              hidden_units=hid_units,
              num_heads=n_heads,
              attn_drop=0.6,
              ftr_drop=0.6,
              activation=activation,
              residual=residual)
    _cell_graph_executor.compile(net, input_data, biases)
