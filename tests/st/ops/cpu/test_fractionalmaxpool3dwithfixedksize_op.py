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


class NetFractionalMaxPool3DWithFixedKsize(nn.Cell):
    def __init__(self, ksize, output_shape):
        super(NetFractionalMaxPool3DWithFixedKsize, self).__init__()
        self.fractional_max_pool_3d_with_fixed_ksize = ops.FractionalMaxPool3DWithFixedKsize(
            ksize, output_shape)

    def construct(self, x, random_sapmples):
        return self.fractional_max_pool_3d_with_fixed_ksize(x, random_sapmples)


class NetFractionalMaxPool3DGradWithFixedKsize(nn.Cell):
    def __init__(self):
        super(NetFractionalMaxPool3DGradWithFixedKsize, self).__init__()
        self.fractional_max_pool_3d_grad_with_fixed_ksize = grad_ops.FractionalMaxPool3DGradWithFixedKsize()

    def construct(self, origin_input, out_backprop, argmax):
        return self.fractional_max_pool_3d_grad_with_fixed_ksize(origin_input, out_backprop, argmax)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fractionalmaxpool3dwithfixedksize():
    """
    Feature: FractionalMaxPool3DWithFixedKsize
    Description: Test of input
    Expectation: The results are as expected
    """
    context_mode_types = [context.GRAPH_MODE, context.PYNATIVE_MODE]
    types_input1 = [np.float16, np.float32, np.int32, np.int64]
    types_input2 = [np.float16, np.float32]
    for context_mode_type in context_mode_types:
        context.set_context(mode=context_mode_type, device_target='CPU')
        for type_input1 in types_input1:
            for type_input2 in types_input2:
                x_np = np.array([i+1 for i in range(64)]
                                ).reshape([1, 1, 4, 4, 4]).astype(type_input1)
                x_ms = Tensor(x_np)
                x_dyn = Tensor(shape=(1, 1, None, None, None),
                               dtype=x_ms.dtype)
                random_samples = Tensor(np.array([0.5, 0.5, 0.8]).reshape(
                    [1, 1, 3]).astype(type_input2))
                ksize = (1, 1, 1)
                output_shape = (2, 2, 3)
                net = NetFractionalMaxPool3DWithFixedKsize(ksize, output_shape)
                net.set_inputs(x_dyn, random_samples)
                output_ms, argmax = net(x_ms, random_samples)
                expect_output = np.array([[[[[1, 2, 4], [13, 14, 16]],
                                            [[49, 50, 52], [61, 62, 64]]]]]).astype(type_input1)
                expect_output_argmax = np.array([[[[[0, 1, 3], [12, 13, 15]],
                                                   [[48, 49, 51], [60, 61, 63]]]]]).astype(type_input2)
                assert np.allclose(output_ms.asnumpy(), expect_output)
                assert np.allclose(argmax.asnumpy(), expect_output_argmax)
