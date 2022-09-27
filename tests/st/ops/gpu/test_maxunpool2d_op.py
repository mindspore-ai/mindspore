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


class NetMaxUnpool2DFourD(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool2DFourD, self).__init__()
        self.maxunpool2d_fun = ops.MaxUnpool2D(ksize=(3, 2), strides=(3, 2), pads=0, data_format='NCHW')

    def construct(self, x, indices):
        return self.maxunpool2d_fun(x, indices)


class NetMaxUnpool2DGradFourD(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool2DGradFourD, self).__init__()
        self.maxunpool2d_grad = grad_ops.MaxUnpool2DGrad(ksize=(1, 1, 3, 2), strides=(1, 1, 3, 2), pads=(1, 1, 0, 0),
                                                         data_format='NCHW')

    def construct(self, x, grad, indices):
        return self.maxunpool2d_grad(x, grad, indices)


class NetMaxUnpool2DFourDNHWC(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool2DFourDNHWC, self).__init__()
        self.maxunpool2d_fun = ops.MaxUnpool2D(ksize=(3, 2), strides=(3, 2), pads=0, data_format='NHWC')

    def construct(self, x, indices):
        return self.maxunpool2d_fun(x, indices)


class NetMaxUnpool2DGradFourDNHWC(nn.Cell):
    def __init__(self):
        super(NetMaxUnpool2DGradFourDNHWC, self).__init__()
        self.maxunpool2d_grad = grad_ops.MaxUnpool2DGrad(ksize=(1, 1, 3, 2), strides=(1, 1, 3, 2), pads=(1, 1, 0, 0),
                                                         data_format='NHWC')

    def construct(self, x, grad, indices):
        return self.maxunpool2d_grad(x, grad, indices)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxunpool2d_4dinput_graph():
    """
    Feature: MaxUnpool2d 4dinput graph
    Description: 4dinput graph
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    indices_type = [np.int32, np.int64]
    inout_types = [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]
    for indices_type_i in indices_type:
        for inout_type_i in inout_types:
            x = Tensor(np.array([[[[10, 12], [22, 24]],
                                  [[34, 36], [46, 48]]],
                                 [[[58, 60], [70, 72]],
                                  [[82, 84], [94, 96]]]]).astype(inout_type_i))
            indices = Tensor(np.array([[[[9, 11], [21, 23]],
                                        [[9, 11], [21, 23]]],
                                       [[[9, 11], [21, 23]],
                                        [[9, 11], [21, 23]]]]).astype(indices_type_i))
            maxunpool2d = NetMaxUnpool2DFourD()
            y = maxunpool2d(x, indices)
            output_type = y.asnumpy().dtype
            expect_result = Tensor(np.array([[[[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 10, 0, 12], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 22, 0, 24]],
                                              [[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 34, 0, 36], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 46, 0, 48]]],
                                             [[[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 58, 0, 60], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 70, 0, 72]],
                                              [[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 82, 0, 84], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 94, 0, 96]]]]).astype(inout_type_i))
            assert np.allclose(expect_result.asnumpy(), y.asnumpy())
            assert output_type == inout_type_i

            maxunpoo2dgrad = NetMaxUnpool2DGradFourD()
            grad = Tensor(np.array([i+1 for i in range(4*24)]).reshape([2, 2, 6, 4]).astype(inout_type_i))
            output_grad = maxunpoo2dgrad(x, grad, indices)
            output_grad_type = output_grad.asnumpy().dtype
            expect_output_grad = Tensor(np.array([[[[10, 12], [22, 24]],
                                                   [[34, 36], [46, 48]]],
                                                  [[[58, 60], [70, 72]],
                                                   [[82, 84], [94, 96]]]]).astype(inout_type_i))
            assert np.allclose(expect_output_grad.asnumpy(), output_grad.asnumpy())
            assert output_grad_type == inout_type_i


def test_maxunpool2d_4dinput_pynative():
    """
    Feature: MaxUnpool2d 4dinput pynative
    Description: 4dinput pynative
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    indices_type = [np.int32, np.int64]
    inout_types = [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]
    for indices_type_i in indices_type:
        for inout_type_i in inout_types:
            x = Tensor(np.array([[[[10, 12], [22, 24]],
                                  [[34, 36], [46, 48]]],
                                 [[[58, 60], [70, 72]],
                                  [[82, 84], [94, 96]]]]).astype(inout_type_i)).transpose(0, 2, 3, 1)
            indices = Tensor(np.array([[[[9, 11], [21, 23]],
                                        [[9, 11], [21, 23]]],
                                       [[[9, 11], [21, 23]],
                                        [[9, 11], [21, 23]]]]).astype(indices_type_i)).transpose(0, 2, 3, 1)
            maxunpool2d = NetMaxUnpool2DFourDNHWC()
            y = maxunpool2d(x, indices)
            output_type = y.asnumpy().dtype
            expect_result = Tensor(np.array([[[[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 10, 0, 12], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 22, 0, 24]],
                                              [[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 34, 0, 36], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 46, 0, 48]]],
                                             [[[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 58, 0, 60], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 70, 0, 72]],
                                              [[0, 0, 0, 0], [0, 0, 0, 0],
                                               [0, 82, 0, 84], [0, 0, 0, 0],
                                               [0, 0, 0, 0], [0, 94, 0, 96]]]])
                                   .astype(inout_type_i)).transpose(0, 2, 3, 1)
            assert np.allclose(expect_result.asnumpy(), y.asnumpy())
            assert output_type == inout_type_i

            maxunpoo2dgrad = NetMaxUnpool2DGradFourDNHWC()
            grad = Tensor(np.array([i+1 for i in range(4*24)]).reshape([2, 2, 6, 4])
                          .astype(inout_type_i)).transpose(0, 2, 3, 1)
            output_grad = maxunpoo2dgrad(x, grad, indices)
            output_grad_type = output_grad.asnumpy().dtype
            expect_output_grad = Tensor(np.array([[[[10, 12], [22, 24]],
                                                   [[34, 36], [46, 48]]],
                                                  [[[58, 60], [70, 72]],
                                                   [[82, 84], [94, 96]]]]).astype(inout_type_i)).transpose(0, 2, 3, 1)
            assert np.allclose(expect_output_grad.asnumpy(), output_grad.asnumpy())
            assert output_grad_type == inout_type_i
