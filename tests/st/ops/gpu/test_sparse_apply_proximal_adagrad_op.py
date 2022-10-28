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

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):

    def __init__(self, var, accum, lr, l1, l2):
        super(Net, self).__init__()
        self.sparse_apply_proximal_adagrad = P.SparseApplyProximalAdagrad()
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.lr = lr
        self.l1 = l1
        self.l2 = l2

    def construct(self, grad, indices):
        out = self.sparse_apply_proximal_adagrad(self.var, self.accum, self.lr,
                                                 self.l1, self.l2, grad,
                                                 indices)
        return out


def add_testcase(var, accum, lr, l1, l2, grad, indices):
    net = Net(var, accum, lr, l1, l2)
    return net(grad, indices)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_dyn_shape():
    """
    Feature: test SparseApplyProximalAdagrad ops in gpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    var = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    accum = Tensor(np.zeros(9).reshape(3, 3).astype(np.float32))
    lr = 1.0
    l1 = 1.0
    l2 = 0.0
    net = Net(var, accum, lr, l1, l2)
    grad_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(grad_dyn, indices_dyn)

    grad = Tensor(np.ones(9).reshape(3, 3).astype(np.float32) * 8)
    indices = Tensor(np.array([1, 0, 2], np.int32))
    output1, output2 = net(grad, indices)

    expect_shapes = [(3, 3), (3, 3)]
    assert output1.asnumpy().shape == expect_shapes[0]
    assert output2.asnumpy().shape == expect_shapes[1]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_small_shape_input_update():
    var = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    accum = Tensor(np.zeros(9).reshape(3, 3).astype(np.float32))
    lr = 1.0
    l1 = 1.0
    l2 = 0.0
    grad = Tensor(np.ones(9).reshape(3, 3).astype(np.float32) * 8)
    indices = Tensor(np.array([1, 0, 2], np.int32))
    net = Net(var, accum, lr, l1, l2)
    net(grad, indices)
    expect1 = np.array([[-0.875, 0., 0.875], [1.875, 2.875, 3.875],
                        [4.875, 5.875, 6.875]])
    expect2 = np.array([[64., 64., 64.], [64., 64., 64.], [64., 64., 64.]])
    np.testing.assert_array_almost_equal(net.var.data.asnumpy(), expect1)
    np.testing.assert_array_almost_equal(net.accum.data.asnumpy(), expect2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_parameter_lr_l1_l2():
    var = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    accum = Tensor(np.zeros(9).reshape(3, 3).astype(np.float32))
    lr = 100.0
    l1 = 34.0
    l2 = 16.0
    grad = Tensor(np.ones(9).reshape(3, 3).astype(np.float32) * 8)
    indices = Tensor(np.array([1, 0, 2], np.int32))
    output1, output2 = add_testcase(var, accum, lr, l1, l2, grad, indices)
    expect1 = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    expect2 = np.array([[64., 64., 64.], [64., 64., 64.], [64., 64., 64.]])
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect1)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_with_np_arange():
    var = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    accum = Tensor(np.arange(63, 72).reshape(3, 3).astype(np.float32))
    lr = 1.0
    l1 = 1.0
    l2 = 2.0
    grad = Tensor(np.arange(34, 43).reshape(3, 3).astype(np.float32) * 8)
    indices = Tensor(np.array([2, 1, 0], np.int32))
    output1, output2 = add_testcase(var, accum, lr, l1, l2, grad, indices)
    expect1 = np.array([[-0.99038047, 0., 0.9914129],
                        [1.9836018, 2.9774926, 3.9716945],
                        [4.9603353, 5.9543643, 6.948723]])
    expect2 = np.array([[102463., 107648., 112961.], [87682., 92483., 97412.],
                        [74053., 78470., 83015.]])
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect1)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_large_shape():
    var = Tensor(np.arange(24).reshape((2, 3, 4)).astype(np.float32))
    accum = Tensor(np.arange(34, 58).reshape((2, 3, 4)).astype(np.float32))
    lr = 1.0
    l1 = 1.0
    l2 = 2.0
    grad = Tensor(np.ones(24).reshape((2, 3, 4)).astype(np.float32) * 2)
    indices = Tensor(np.arange(2).astype(np.int32))
    output1, output2 = add_testcase(var, accum, lr, l1, l2, grad, indices)
    #expected outputs are from Dchip
    expect1 = np.array([[[-0.12248275, 0.39357165, 1.1591142, 1.9289699],
                         [2.7029436, 3.4808538, 4.2625313, 5.0478177],
                         [5.836565, 6.6286335, 7.423894, 8.222222]],
                        [[9.023503, 9.82763, 10.634497, 11.444007],
                         [12.256072, 13.0706005, 13.887513, 14.706733],
                         [15.528182, 16.35179, 17.177492, 18.005226]]])
    expect2 = np.array([[[38., 39., 40., 41.], [42., 43., 44., 45.],
                         [46., 47., 48., 49.]],
                        [[50., 51., 52., 53.], [54., 55., 56., 57.],
                         [58., 59., 60., 61.]]])
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect1)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect2)
