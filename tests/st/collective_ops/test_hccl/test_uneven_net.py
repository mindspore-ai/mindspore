# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test uneven net"""

#msrun --worker_num=2 --local_worker_num=2 --master_port=10923 --bind_core True --join True --cluster_time_out=800  pytest -sv --disable-warnings test_uneven_net.py
import mindspore.nn as nn
from mindspore.graph.api import _cell_graph_executor
from mindspore.communication.management import init, get_rank
from mindspore.nn import Dense
from mindspore.nn import Momentum
from mindspore.nn import ReLU
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops.operations.comm_ops import AllGatherV, ReduceScatterV
from mindspore import Tensor, context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")
init()
rank = get_rank()


class UNEVEN_Net(nn.Cell):
    def __init__(self, input_channel, out_channel):
        super(UNEVEN_Net, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.allgatherv = AllGatherV()
        self.reducev = ReduceScatterV()
        self.relu = ReLU()
    def construct(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = P.Reshape()(x, (-1,))
        x = self.allgatherv(x, [3, 3])
        x = self.reducev(x, [3, 3])
        x = self.relu(x)
        x = P.Reshape()(x, (-1, 1))
        return x


def test_uneven():
    """
    Feature: test 'AllGatherV' and 'ReduceScatterV' communication operator.
    Description: test 'AllGatherV' and 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """

    input_tensor = Tensor([[0, 1, 2.], [3, 4, 5], [6, 7, 8]])
    label_tensor = Tensor([[1.0], [2], [3]])
    network = UNEVEN_Net(3, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _cell_graph_executor.compile(network, input_tensor, label_tensor)
