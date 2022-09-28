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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.nn_ops as ops
import mindspore.ops.operations._grad_ops as grad_ops


class NetMaxUnpool3DFiveD(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool3DFiveD, self).__init__()
        self.maxunpool3d_fun = ops.MaxUnpool3D(ksize=(2, 2, 2), strides=(2, 2, 2), pads=(0, 0, 0), data_format='NCDHW')

    def construct(self, x, indices):
        return self.maxunpool3d_fun(x, indices)


class NetMaxUnpool3DGradFiveD(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool3DGradFiveD, self).__init__()
        self.maxunpool3d_grad = grad_ops.MaxUnpool3DGrad(ksize=(1, 1, 2, 2, 2), strides=(1, 1, 2, 2, 2),
                                                         pads=(1, 1, 0, 0, 0), data_format='NCDHW')

    def construct(self, x, grad, indices):
        return self.maxunpool3d_grad(x, grad, indices)


class NetMaxUnpool3DFiveDNDHWC(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool3DFiveDNDHWC, self).__init__()
        self.maxunpool3d_fun = ops.MaxUnpool3D(ksize=(2, 2, 2), strides=(2, 2, 2), pads=(0, 0, 0), data_format='NDHWC')

    def construct(self, x, indices):
        return self.maxunpool3d_fun(x, indices)


class NetMaxUnpool3DGradFiveDNDHWC(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool3DGradFiveDNDHWC, self).__init__()
        self.maxunpool3d_grad = grad_ops.MaxUnpool3DGrad(ksize=(1, 1, 2, 2, 2), strides=(1, 1, 2, 2, 2),
                                                         pads=(1, 1, 0, 0, 0), data_format='NDHWC')

    def construct(self, x, grad, indices):
        return self.maxunpool3d_grad(x, grad, indices)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxunpool3d_5dinput_graph():
    """
    Feature: MaxUnpool3D
    Description: Test of 5D input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    indices_type = [np.int32, np.int64]
    inout_types = [np.int8, np.int16, np.int32, np.int64, np.uint8,
                   np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]
    for indices_type_i in indices_type:
        for inout_type_i in inout_types:
            x = Tensor(np.array([[[[[8]]], [[[16]]]],
                                 [[[[24]]], [[[32]]]]]).astype(inout_type_i))
            indices = Tensor(np.array([[[[[7]]], [[[7]]]],
                                       [[[[7]]], [[[7]]]]]).astype(indices_type_i))
            maxunpool3d = NetMaxUnpool3DFiveD()
            y = maxunpool3d(x, indices)
            output_type = y.asnumpy().dtype
            expect_result = Tensor(np.array([[[[[0, 0], [0, 0]],
                                               [[0, 0], [0, 8]]],
                                              [[[0, 0], [0, 0]],
                                               [[0, 0], [0, 16]]]],
                                             [[[[0, 0], [0, 0]],
                                               [[0, 0], [0, 24]]],
                                              [[[0, 0], [0, 0]],
                                               [[0, 0], [0, 32]]]]]).astype(inout_type_i))
            assert np.allclose(expect_result.asnumpy(), y.asnumpy())
            assert output_type == inout_type_i

            maxunpoo2dgrad = NetMaxUnpool3DGradFiveD()
            grad = Tensor(np.array([i+1 for i in range(32)]).reshape([2, 2, 2, 2, 2]).astype(inout_type_i))
            output_grad = maxunpoo2dgrad(x, grad, indices)
            output_grad_type = output_grad.asnumpy().dtype
            expect_output_grad = Tensor(np.array([[[[[8]]], [[[16]]]],
                                                  [[[[24]]], [[[32]]]]]).astype(inout_type_i))
            assert np.allclose(expect_output_grad.asnumpy(), output_grad.asnumpy())
            assert output_grad_type == inout_type_i


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxunpool3d_5dinput_pynative():
    """
    Feature: MaxUnpool3D
    Description: Test of 5D input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    indices_type = [np.int32, np.int64]
    inout_types = [np.int8, np.int16, np.int32, np.int64, np.uint8,
                   np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]
    for indices_type_i in indices_type:
        for inout_type_i in inout_types:
            x = Tensor(np.array([[[[[8]]], [[[16]]]],
                                 [[[[24]]], [[[32]]]]]).astype(inout_type_i)).transpose(0, 2, 3, 4, 1)
            indices = Tensor(np.array([[[[[7]]], [[[7]]]],
                                       [[[[7]]], [[[7]]]]]).astype(indices_type_i)).transpose(0, 2, 3, 4, 1)
            maxunpool3d = NetMaxUnpool3DFiveDNDHWC()
            y = maxunpool3d(x, indices)
            output_type = y.asnumpy().dtype
            expect_result = Tensor(np.array([[[[[0, 0], [0, 0]],
                                               [[0, 0], [0, 8]]],
                                              [[[0, 0], [0, 0]],
                                               [[0, 0], [0, 16]]]],
                                             [[[[0, 0], [0, 0]],
                                               [[0, 0], [0, 24]]],
                                              [[[0, 0], [0, 0]],
                                               [[0, 0], [0, 32]]]]]).astype(inout_type_i)).transpose(0, 2, 3, 4, 1)
            assert np.allclose(expect_result.asnumpy(), y.asnumpy())
            assert output_type == inout_type_i

            maxunpoo2dgrad = NetMaxUnpool3DGradFiveDNDHWC()
            grad = Tensor(np.array([i + 1 for i in range(32)]).reshape([2, 2, 2, 2, 2])
                          .astype(inout_type_i)).transpose(0, 2, 3, 4, 1)
            output_grad = maxunpoo2dgrad(x, grad, indices)
            output_grad_type = output_grad.asnumpy().dtype
            expect_output_grad = Tensor(np.array([[[[[8]]], [[[16]]]],
                                                  [[[[24]]], [[[32]]]]]).astype(inout_type_i)).transpose(0, 2, 3, 4, 1)
            assert np.allclose(expect_output_grad.asnumpy(), output_grad.asnumpy())
            assert output_grad_type == inout_type_i
