# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import ops as P
from mindspore import context, Tensor, nn
from mindspore.ops.operations import _grad_ops as G

import numpy as np
import pytest


class LogsoftmaxGradNet(nn.Cell):
    def __init__(self, axis):
        super(LogsoftmaxGradNet, self).__init__()
        self.op = G.LogSoftmaxGrad(axis)

    def construct(self, out, dout):
        res = self.op(out, dout)
        return res


class LogsoftmaxGradDynamicRankNet(nn.Cell):
    def __init__(self, axis):
        super(LogsoftmaxGradDynamicRankNet, self).__init__()
        self.op = G.LogSoftmaxGrad(axis)
        self.reduce_sum = P.ReduceSum(keep_dims=False)

    def construct(self, out, dout, dyn_reduce_axis):
        out = self.reduce_sum(out, dyn_reduce_axis)
        dout = self.reduce_sum(dout, dyn_reduce_axis)
        res = self.op(out, dout)
        return res


def case_logsoftmax_grad_dyn(mode, device_target):
    context.set_context(mode=mode, device_target=device_target)
    net = LogsoftmaxGradNet(0)
    x = np.random.randn(16, 32).astype(np.float32)
    y = np.random.randn(16, 32).astype(np.float32)
    static_out = net(Tensor(x), Tensor(y)).asnumpy()

    dyn_x = Tensor(shape=[16, None], dtype=ms.float32)
    dyn_y = Tensor(shape=[16, None], dtype=ms.float32)
    dyn_net = LogsoftmaxGradNet(0)
    dyn_net.set_inputs(dyn_x, dyn_y)
    dyn_out = dyn_net(Tensor(x), Tensor(y)).asnumpy()
    assert np.allclose(dyn_out, static_out, 1e-3, 1e-3)

    dyn_rank_net = LogsoftmaxGradDynamicRankNet(0)
    reduce_axis = np.array([2], dtype=np.int64)
    dyn_x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    dyn_y = Tensor(shape=[16, None, 1], dtype=ms.float32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=ms.int64)
    dyn_rank_net.set_inputs(dyn_x, dyn_y, dyn_reduce_axis)
    dyn_rank_out = dyn_rank_net(Tensor(np.expand_dims(x, -1)),
                                Tensor(np.expand_dims(y, -1)), Tensor(reduce_axis)).asnumpy()
    assert np.allclose(dyn_rank_out, static_out, 1e-3, 1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_logsoftmax_grad_dyn_gpu():
    """
    Feature: Test LogSoftmaxGrad op on GPU.
    Description: The input shape is dynamic
    Expectation: Assert the result is equal the static result.
    """
    case_logsoftmax_grad_dyn(context.GRAPH_MODE, "GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_logsoftmax_grad_dyn_cpu():
    """
    Feature: Test LogSoftmaxGrad op on CPU.
    Description: The input shape is dynamic
    Expectation: Assert the result is equal the static result.
    """
    case_logsoftmax_grad_dyn(context.GRAPH_MODE, "CPU")
