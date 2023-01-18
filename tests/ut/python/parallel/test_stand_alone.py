# Copyright 2023 Huawei Technologies Co., Ltd
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


import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import prim_attr_register, PrimitiveWithInfer


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class VirtualNodeGrad(PrimitiveWithInfer):
    """ VirtualLossGrad definition """

    @prim_attr_register
    def __init__(self):
        """init VirtualLossGrad"""

    def __call__(self, x, out, dout):
        raise NotImplementedError

    def infer_shape(self, x_shape, out_shape, dout_shape):
        return x_shape

    def infer_dtype(self, x_dtype, out_dtype, dout_dtype):
        return x_dtype


class VirtualNode(PrimitiveWithInfer):
    """ VirtualLoss definition """

    @prim_attr_register
    def __init__(self):
        """init VirtualLoss"""

    def __call__(self, x):
        raise NotImplementedError

    def get_bprop(self):
        loss_grad = VirtualNodeGrad()

        def bprop(x, out, dout):
            dx = loss_grad(x, out, dout)
            return (dx,)

        return bprop

    def infer_shape(self, x_shape):
        return [1]

    def infer_dtype(self, x_dtype):
        return x_dtype


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualNode()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def test_two_matmul():
    '''
    Feature: test StandAloneInfo
    Description: In SemiAuto mode, if there is no strategy and can't use BatchParallelInfo, use StandAloneInfo
    Expectation: success
    '''
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, full_batch=True)
    strategy1 = ((1, 2), (2, 2))
    strategy2 = ((1, 2), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([1, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    b = Tensor(np.ones([128, 128]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)
