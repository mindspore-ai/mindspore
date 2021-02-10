# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.ops import composite as C

class StridedSliceNet(nn.Cell):
    def __init__(self, begin, end, stride, begin_mask=0, end_mask=0, ellipsis_mask=0):
        super(StridedSliceNet, self).__init__()
        self.begin = begin
        self.end = end
        self.strides = stride
        self.slice = P.StridedSlice(begin_mask, end_mask, ellipsis_mask)

    def construct(self, x):
        return self.slice(x, self.begin, self.end, self.strides)

class GradData(nn.Cell):
    def __init__(self, network):
        super(GradData, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, x):
        return self.grad(self.network)(x)


def strided_slice_grad(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    x = Tensor(np.arange(0, 2*3*4*5).reshape(2, 3, 4, 5).astype(nptype))
    net = StridedSliceNet((1, 0, 0, 2), (2, 2, 2, 4), (1, 1, 1, 1))
    dx = GradData(net)(x)
    expect = np.array([[[[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]],


                       [[[0., 0., 1., 1., 0.],
                         [0., 0., 1., 1., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 1., 1., 0.],
                         [0., 0., 1., 1., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]]]).astype(nptype)
    assert np.allclose(dx[0].asnumpy(), expect)

    net = StridedSliceNet((1, 0, 0, 5), (2, 2, 2, 1), (1, 1, 1, -2))
    dx = GradData(net)(x)
    expect = np.array([[[[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]],


                       [[[0., 0., 1., 0., 1.],
                         [0., 0., 1., 0., 1.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 1., 0., 1.],
                         [0., 0., 1., 0., 1.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]]]).astype(nptype)
    assert np.allclose(dx[0].asnumpy(), expect)


    net = StridedSliceNet((1, 0, 0, -1), (2, 2, 2, 1), (1, 1, 1, -1))
    dx = GradData(net)(x)
    expect = np.array([[[[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]],


                       [[[0., 0., 1., 1., 1.],
                         [0., 0., 1., 1., 1.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 1., 1., 1.],
                         [0., 0., 1., 1., 1.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]]]).astype(nptype)
    assert np.allclose(dx[0].asnumpy(), expect)


    net = StridedSliceNet((1, 0, 0, 2), (2, 2, 2, 4), (1, 1, 1, 1),
                          begin_mask=0b1000, end_mask=0b0010, ellipsis_mask=0b0100)
    dx = GradData(net)(x)
    expect = np.array([[[[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]],


                       [[[1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.]],

                        [[1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.]],

                        [[1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 0.]]]]).astype(nptype)
    assert np.allclose(dx[0].asnumpy(), expect)

    x = Tensor(np.arange(0, 3*4*5).reshape(3, 4, 5).astype(np.float32))
    net = StridedSliceNet((1, 0, 0), (2, -3, 3), (1, 1, 3))
    dx = GradData(net)(x)
    expect = np.array([[[0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]],

                       [[1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]],

                       [[0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]]]).astype(nptype)
    assert np.allclose(dx[0].asnumpy(), expect)

    x = Tensor(np.arange(0, 1 * 1 * 1 * 2 * 3 * 4 * 5).reshape(1, 1, 1, 2, 3, 4, 5).astype(nptype))
    net = StridedSliceNet((0, 0, 0, 1, 1, 2, 2), (1, 1, 1, 2, 3, 3, 4), (1, 1, 1, 1, 1, 1, 1))
    dx = GradData(net)(x)
    expect = np.array([[[[[[[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],

                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],

                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]],

                          [[[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],

                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 1., 1., 0.],
                            [0., 0., 0., 0., 0.]],

                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 1., 1., 0.],
                            [0., 0., 0., 0., 0.]]]]]]]).astype(nptype)
    assert np.allclose(dx[0].asnumpy(), expect)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_float64():
    strided_slice_grad(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_float32():
    strided_slice_grad(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_float16():
    strided_slice_grad(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_int64():
    strided_slice_grad(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_int32():
    strided_slice_grad(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_int16():
    strided_slice_grad(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_int8():
    strided_slice_grad(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_uint64():
    strided_slice_grad(np.uint64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_uint32():
    strided_slice_grad(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_uint16():
    strided_slice_grad(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_uint8():
    strided_slice_grad(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_strided_slice_grad_bool():
    strided_slice_grad(np.bool)
