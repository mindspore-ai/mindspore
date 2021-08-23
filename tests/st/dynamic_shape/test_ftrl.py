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
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.nn import TrainOneStepCell
from mindspore.nn.optim import FTRL, LazyAdam
from mindspore.ops import operations as P

context.set_context(enable_sparse=True,
                    mode=context.GRAPH_MODE,
                    device_target="Ascend")

class NetWithSparseGatherV2(nn.Cell):
    def __init__(self):
        super(NetWithSparseGatherV2, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight1")
        self.weight2 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight2")
        self.axis = 0
        self.gather = P.SparseGatherV2()

    def construct(self, indices, label):
        return self.gather(self.weight1, indices, self.axis) + self.weight2

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ftrl_net():
    indices = Tensor(np.array([0, 0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()

    optimizer = FTRL(net.trainable_params(), learning_rate=0.1, weight_decay=0.9, loss_scale=2.0)
    optimizer.target = 'Ascend'
    train_network = TrainOneStepCell(net, optimizer)
    output = train_network(indices, label)
    np.allclose(output.asnumpy(), np.array([[[2, 2]], [[2, 2]], [[2, 2]]]))
    np.allclose(net.weight1.asnumpy(), np.array([[[0.7884067, 0.7884067]],
                                                 [[0.68213105, 0.68213105]],
                                                 [[1.0, 1.0]]]))
    np.allclose(net.weight2.asnumpy(), np.array([[[0.6821311, 0.6821311]],
                                                 [[0.6821311, 0.6821311]],
                                                 [[0.6821311, 0.6821311]]]))

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lazy_adam_net():
    indices = Tensor(np.array([0, 0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()

    optimizer = LazyAdam(net.trainable_params(), learning_rate=0.1, weight_decay=0.9, loss_scale=2.0)
    optimizer.target = 'Ascend'
    train_network = TrainOneStepCell(net, optimizer)
    output = train_network(indices, label)
    np.allclose(output.asnumpy(), np.array([[[2, 2]], [[2, 2]], [[2, 2]]]))
    np.allclose(net.weight1.asnumpy(), np.array([[[0.9, 0.9]], [[0.9, 0.9]], [[1.0, 1.0]]]))
    np.allclose(net.weight2.asnumpy(), np.array([[[0.9, 0.9]], [[0.9, 0.9]], [[0.9, 0.9]]]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lazy_adam_net_sparse():
    indices = Tensor(np.array([0, 0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()

    optimizer = LazyAdam(net.trainable_params(), learning_rate=0.1, weight_decay=0.9, loss_scale=2.0)
    # will use sparse_opt in LazyAdam
    optimizer.target = 'CPU'
    train_network = TrainOneStepCell(net, optimizer)
    output = train_network(indices, label)
    np.allclose(output.asnumpy(), np.array([[[2, 2]], [[2, 2]], [[2, 2]]]))
    np.allclose(net.weight1.asnumpy(), np.array([[[0.9, 0.9]], [[0.9, 0.9]], [[1.0, 1.0]]]))
    np.allclose(net.weight2.asnumpy(), np.array([[[0.9, 0.9]], [[0.9, 0.9]], [[0.9, 0.9]]]))
