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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
import mindspore.common.dtype as mstype
from mindspore.ops import composite as C
from mindspore.nn.reinforcement._tensors_queue import TensorsQueue


class TensorsQueueNet(nn.Cell):
    def __init__(self, dtype, shapes, size=0, name="q"):
        super(TensorsQueueNet, self).__init__()
        self.tq = TensorsQueue(dtype, shapes, size, name)

    def construct(self, grads):
        self.tq.put(grads)
        self.tq.put(grads)
        size_before = self.tq.size()
        ans = self.tq.pop()
        size_after = self.tq.size()
        self.tq.clear()
        self.tq.close()
        return ans, size_before, size_after


class SourceNet(nn.Cell):
    '''Source net'''
    def __init__(self):
        super(SourceNet, self).__init__()
        self.a = Parameter(Tensor(0.5, mstype.float32), name="a")
        self.dense = nn.Dense(in_channels=4, out_channels=1, weight_init=0)

    def construct(self, data):
        d = self.dense(data)
        out = d + self.a
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensorsqueue_gpu():
    """
    Feature: TensorsQueue gpu TEST.
    Description: Test the function write, read, stack, clear, close in both graph and pynative mode.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_data = Tensor(np.arange(8).reshape(2, 4), mstype.float32)
    net = SourceNet()
    weight = ParameterTuple(net.trainable_params())
    grad = C.GradOperation(get_by_list=True, sens_param=False)
    _ = net(input_data)
    grads = grad(net, weight)(input_data)
    shapes = []
    for i in grads:
        shapes.append(i.shape)
    tq = TensorsQueueNet(dtype=mstype.float32, shapes=shapes, size=5, name="tq")
    ans, size_before, size_after = tq(grads)
    assert np.allclose(size_before, 2)
    assert np.allclose(size_after, 1)
    assert np.allclose(ans[0], 2.0)
    assert np.allclose(ans[1].asnumpy(), [[4.0, 6.0, 8.0, 10.0]])
    assert np.allclose(ans[2].asnumpy(), [2.0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensorsqueue_cpu():
    """
    Feature: TensorsQueue cpu TEST.
    Description: Test the function write, read, stack, clear, close in both graph and pynative mode.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    data = Tensor(np.arange(8).reshape(2, 4), mstype.float32)
    net = SourceNet()
    weight = ParameterTuple(net.trainable_params())
    grad = C.GradOperation(get_by_list=True, sens_param=False)
    _ = net(data)
    grads = grad(net, weight)(data)
    shapes = []
    for i in grads:
        shapes.append(i.shape)
    tq_cpu = TensorsQueueNet(dtype=mstype.float32, shapes=shapes, size=5, name="tq")
    ans, size_before, size_after = tq_cpu(grads)
    assert np.allclose(ans[0], 2.0)
    assert np.allclose(ans[1].asnumpy(), [[4.0, 6.0, 8.0, 10.0]])
    assert np.allclose(ans[2].asnumpy(), [2.0])
    assert np.allclose(size_before, 2)
    assert np.allclose(size_after, 1)
