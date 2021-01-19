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

import mindspore.ops.operations as P
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops import composite as C


class MaxPoolWithArgMax_Net(Cell):
    def __init__(self, padding, ksize, strides):
        super(MaxPoolWithArgMax_Net, self).__init__()
        self.maxpool_with_argmax = P.MaxPoolWithArgmax(pad_mode=padding, kernel_size=ksize, strides=strides)

    def construct(self, input_data):
        output, argmax = self.maxpool_with_argmax(input_data)
        return output, argmax


class Grad(Cell):
    def __init__(self, network, argmax):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens = (Tensor(np.ones(argmax.shape).astype(np.float32)),
                     Tensor(np.ones(argmax.shape).astype(np.int32)))

    def construct(self, input_data):
        gout = self.grad(self.network)(input_data, self.sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_forward_backward():
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    expect_output = np.array([[[[5, 6, 7, 7],
                                [9, 10, 11, 11],
                                [9, 10, 11, 11]],
                               [[17, 18, 19, 19],
                                [21, 22, 23, 23],
                                [21, 22, 23, 23]],
                               [[29, 30, 31, 31],
                                [33, 34, 35, 35],
                                [33, 34, 35, 35]]]]).astype(np.float32)
    expect_argmax = np.array([[[[5, 6, 7, 7],
                                [9, 10, 11, 11],
                                [9, 10, 11, 11]],
                               [[17, 18, 19, 19],
                                [21, 22, 23, 23],
                                [21, 22, 23, 23]],
                               [[29, 30, 31, 31],
                                [33, 34, 35, 35],
                                [33, 34, 35, 35]]]]).astype(np.int32)
    expect_dx = np.array([[[[0, 0, 0, 0],
                            [0, 1, 1, 2],
                            [0, 2, 2, 4]],
                           [[0, 0, 0, 0],
                            [0, 1, 1, 2],
                            [0, 2, 2, 4]],
                           [[0, 0, 0, 0],
                            [0, 1, 1, 2],
                            [0, 2, 2, 4]]]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = MaxPoolWithArgMax_Net(padding="SAME", ksize=2, strides=1)
    output_tensor, argmax_tensor = net(Tensor(x))
    assert output_tensor.shape == expect_output.shape
    assert argmax_tensor.shape == expect_argmax.shape

    error = np.ones(shape=expect_output.shape) * 1.0e-5
    diff_output = output_tensor.asnumpy() - expect_output
    assert np.all(diff_output < error)

    net_grad = Grad(net, argmax_tensor)
    dx = net_grad(Tensor(x))[0].asnumpy()
    assert dx.shape == expect_dx.shape
    diff = dx - expect_dx
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool_with_argmax_2d():
    x = Tensor(np.array([[[
        [0, 1, 2, 3, -4, -5],
        [6, 7, 8, 9, -10, -11],
        [12, 13, 14, -15, -16, -17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float32))
    expect_result = (np.array([[[
        [7, 9, -4],
        [19, 21, 23],
        [31, 33, 35]
    ]]]))
    expect_result2 = (np.array([[[
        [14, 14, -4],
        [26, 28, 29],
        [32, 34, 35]
    ]]]))
    expect_index_result = (np.array([[[
        [7, 9, 4],
        [19, 21, 23],
        [31, 33, 35]
    ]]]))
    expect__index_result2 = (np.array([[[
        [14, 14, 4],
        [26, 28, 29],
        [32, 34, 35]
    ]]]))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    maxpool2d = MaxPoolWithArgMax_Net(padding="VALID", ksize=2, strides=2)
    maxpool2d2 = MaxPoolWithArgMax_Net(padding="SAME", ksize=3, strides=2)
    output2, index2 = maxpool2d2(x)
    output, index = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()
    assert (index.asnumpy() == expect_index_result).all()
    assert (index2.asnumpy() == expect__index_result2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    maxpool2d = MaxPoolWithArgMax_Net(padding="VALID", ksize=2, strides=2)
    maxpool2d2 = MaxPoolWithArgMax_Net(padding="SAME", ksize=3, strides=2)
    output2, index2 = maxpool2d2(x)
    output, index = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()
    assert (index.asnumpy() == expect_index_result).all()
    assert (index2.asnumpy() == expect__index_result2).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool_with_argmax_2d_fp16():
    x = Tensor(np.array([[[
        [0, 1, 2, 3, -4, -5],
        [6, 7, 8, 9, -10, -11],
        [12, 13, 14, -15, -16, -17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float16))
    expect_result = (np.array([[[
        [7, 9, -4],
        [19, 21, 23],
        [31, 33, 35]
    ]]]))
    expect_result2 = (np.array([[[
        [14, 14, -4],
        [26, 28, 29],
        [32, 34, 35]
    ]]]))
    expect_index_result = (np.array([[[
        [7, 9, 4],
        [19, 21, 23],
        [31, 33, 35]
    ]]]))
    expect__index_result2 = (np.array([[[
        [14, 14, 4],
        [26, 28, 29],
        [32, 34, 35]
    ]]]))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    maxpool2d = MaxPoolWithArgMax_Net(padding="VALID", ksize=2, strides=2)
    maxpool2d2 = MaxPoolWithArgMax_Net(padding="SAME", ksize=3, strides=2)
    output2, index2 = maxpool2d2(x)
    output, index = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()
    assert (index.asnumpy() == expect_index_result).all()
    assert (index2.asnumpy() == expect__index_result2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    maxpool2d = MaxPoolWithArgMax_Net(padding="VALID", ksize=2, strides=2)
    maxpool2d2 = MaxPoolWithArgMax_Net(padding="SAME", ksize=3, strides=2)
    output2, index2 = maxpool2d2(x)
    output, index = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()
    assert (index.asnumpy() == expect_index_result).all()
    assert (index2.asnumpy() == expect__index_result2).all()
