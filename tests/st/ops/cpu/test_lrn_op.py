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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from mindspore.common import dtype as ms_type


class LrnNet(nn.Cell):
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS"):
        super(LrnNet, self).__init__()
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.norm_region = norm_region
        self.lrn = P.LRN(depth_radius, bias, alpha, beta, norm_region)

    def construct(self, input_x):
        output = self.lrn(input_x)
        return output


class LrnVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(LrnVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x)


def lrn_np_bencmark(data_type):
    """
    Feature: generate a lrn numpy benchmark.
    Description: The input shape need to match to output shape.
    Expectation: match to np mindspore LRN.
    """

    y_exp = np.array([[[[1.6239204, -0.61149347],
                        [-0.5279556, -1.0724881]],
                       [[0.86518127, -2.3005495],
                        [1.7440975, -0.760866]],
                       [[0.31895563, -0.2492632],
                        [1.4615093, -2.059218]]]]).astype(data_type)
    return y_exp


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_lrn(data_type):
    """
    Feature: Test LRN.
    Description: The input shape need to match to output shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_data = np.array([[[[1.6243454, -0.6117564],
                             [-0.5281718, -1.0729686]],
                            [[0.86540765, -2.3015387],
                             [1.7448118, -0.7612069]],
                            [[0.3190391, -0.24937038],
                             [1.4621079, -2.0601406]]]]).astype(data_type)
    loss = 1e-6
    if data_type == np.float16:
        loss = 1e-3
    benchmark_output = lrn_np_bencmark(data_type)
    lrn = LrnNet(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
    output = lrn(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = lrn(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lrn_vmap():
    """
    Feature: Test LRN Vmap on CPU.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data_type = np.float32
    loss = 1e-6
    input_x = np.array([[[[[1.6243454, -0.6117564],
                           [-0.5281718, -1.0729686]],
                          [[0.86540765, -2.3015387],
                           [1.7448118, -0.7612069]],
                          [[0.3190391, -0.24937038],
                           [1.4621079, -2.0601406]]]],
                        [[[[1.6243454, -0.6117564],
                           [-0.5281718, -1.0729686]],
                          [[0.86540765, -2.3015387],
                           [1.7448118, -0.7612069]],
                          [[0.3190391, -0.24937038],
                           [1.4621079, -2.0601406]]]]]).astype(data_type)
    benchmark_output = np.array([[[[[1.6239204, -0.61149347],
                                    [-0.5279556, -1.0724881]],
                                   [[0.86518127, -2.3005495],
                                    [1.7440975, -0.760866]],
                                   [[0.31895563, -0.2492632],
                                    [1.4615093, -2.059218]]]],
                                 [[[[1.6239204, -0.61149347],
                                    [-0.5279556, -1.0724881]],
                                   [[0.86518127, -2.3005495],
                                    [1.7440975, -0.760866]],
                                   [[0.31895563, -0.2492632],
                                    [1.4615093, -2.059218]]]]]).astype(data_type)
    lrn = LrnNet(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
    in_axes = 0
    out_axes = 0
    output = LrnVMapNet(lrn, in_axes, out_axes)(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_lrn_dy_shape():
    """
    Feature: Test LRN Dynamic Shape.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    ms_data_type = ms_type.float32
    data_type = np.float32
    # The shape of x is (1, 3, 2, 2)
    x = np.array([[[[1.6243454, -0.6117564],
                    [-0.5281718, -1.0729686]],
                   [[0.86540765, -2.3015387],
                    [1.7448118, -0.7612069]],
                   [[0.3190391, -0.24937038],
                    [1.4621079, -2.0601406]]]]).astype(data_type)
    loss = 1e-6
    benchmark_output = lrn_np_bencmark(data_type)
    lrn = LrnNet(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
    input_dyn = Tensor(shape=[1, 3, 2, None], dtype=ms_data_type)
    lrn.set_inputs(input_dyn)
    output = lrn(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    input_dyn = Tensor(shape=[1, 3, 2, None], dtype=ms_data_type)
    lrn.set_inputs(input_dyn)
    output = lrn(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)
