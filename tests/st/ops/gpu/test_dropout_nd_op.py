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
from operator import mul
from functools import reduce
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from mindspore.common import dtype as ms_type


def check_dropout_nd_by_keep_prob(func_name, input_x, output, output_mask, keep_prob):
    """
    Feature: check mindspore Dropout2D or Dropout3D's output and mask.
    Description: output shape, mask shap and keep_pro will be checked.
    Expectation: match to mindspore Dropout2D or Dropout3D.
    """
    # Check input, output, mask all have same shape
    assert input_x.shape == output.shape == output_mask.shape
    data_type = input_x.dtype
    loss = 1e-6
    if data_type == np.float16:
        loss = 1e-3
    data_shape = input_x.shape
    rank = len(data_shape)
    features = 1
    if func_name == "dropout2d":
        # HW
        features = features * data_shape[-2] * data_shape[-1]
        channels = reduce(mul, data_shape[: rank - 2])
    else:
        # DHW
        features = features * data_shape[-3] * data_shape[-2] * data_shape[-1]
        channels = reduce(mul, data_shape[: rank - 3])
    if keep_prob == 0.0:
        input_x_by_keep_prob = input_x.astype(data_type).reshape(channels, features)
    else:
        input_x_by_keep_prob = (input_x / keep_prob).astype(data_type).reshape(channels, features)
    output_reshape = output.reshape(channels, features)
    mask_reshape = output_mask.reshape(channels, features)
    # Check each channel is entirely True or False and output match to input_x
    for channel in range(channels):
        if np.all(output_reshape[channel] == 0):
            assert int(np.all(mask_reshape[channel])) == 0
        else:
            assert np.all(mask_reshape[channel])
            np.allclose(input_x_by_keep_prob[channel], output_reshape[channel], loss, loss)


class DropoutNdVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(DropoutNdVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x)


class Dropout3DNet(nn.Cell):
    def __init__(self, keep_prob):
        super(Dropout3DNet, self).__init__()
        self.drop = P.Dropout3D(keep_prob)

    def construct(self, x):
        return self.drop(x)


class Dropout2DNet(nn.Cell):
    def __init__(self, keep_prob):
        super(Dropout2DNet, self).__init__()
        self.drop = P.Dropout2D(keep_prob)

    def construct(self, x):
        return self.drop(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("func_name", ["dropout2d", "dropout3d"])
@pytest.mark.parametrize("keep_prob", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("data_type", [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
def test_dropout_nd(func_name, data_type, keep_prob):
    """
    Feature: Test Dropout2D and Dropout3D.
    Description: The input shape is 4d or 5d.
    Expectation: check it by function check_dropout_nd_by_keep_prob.
    """
    if func_name == "dropout2d":
        dropout_nd = Dropout2DNet(keep_prob)
        data_shape = (32, 16, 4, 5)
    else:
        data_shape = (32, 16, 2, 5, 4)
        dropout_nd = Dropout3DNet(keep_prob)
    input_data = np.ones(data_shape).astype(data_type)
    output, mask = dropout_nd(Tensor(input_data))
    context.set_context(mode=context.GRAPH_MODE)
    check_dropout_nd_by_keep_prob(func_name, input_data, output.asnumpy(), mask.asnumpy(), keep_prob)
    context.set_context(mode=context.PYNATIVE_MODE)
    output, mask = dropout_nd(Tensor(input_data))
    check_dropout_nd_by_keep_prob(func_name, input_data, output.asnumpy(), mask.asnumpy(), keep_prob)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("func_name", ["dropout2d", "dropout3d"])
def test_dropout_nd_vmap(func_name):
    """
    Feature: Test dropout2d or dropout3d Vmap on CPU.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data_type = np.float32
    in_axes = 0
    out_axes = 0
    keep_prob = 0.4
    if func_name == "dropout2d":
        data_shape = (5, 32, 16, 4, 5)
        dropout = Dropout2DNet(keep_prob)
    else:
        data_shape = (10, 5, 32, 16, 4, 5)
        dropout = Dropout3DNet(keep_prob)
    input_x = np.ones(data_shape).astype(data_type)
    output, mask = DropoutNdVMapNet(dropout, in_axes, out_axes)(Tensor(input_x))
    check_dropout_nd_by_keep_prob(func_name, input_x, output.asnumpy(), mask.asnumpy(), keep_prob)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("func_name", ["dropout2d", "dropout3d"])
def test_dropout_nd_dy_shape(func_name):
    """
    Feature: Test dropout2d or dropout3d Dynamic Shape.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    ms_data_type = ms_type.float32
    data_type = np.float32
    keep_prob = 0.4
    if func_name == "dropout2d":
        data_shape = (32, 16, 4, 5)
        dropout = Dropout2DNet(keep_prob)
        input_dyn = Tensor(shape=[32, 16, 4, None], dtype=ms_data_type)
    else:
        data_shape = (32, 16, 2, 5, 4)
        dropout = Dropout3DNet(keep_prob)
        input_dyn = Tensor(shape=[32, 16, 2, 5, None], dtype=ms_data_type)

    input_x = np.ones(data_shape).astype(data_type)
    dropout.set_inputs(input_dyn)
    output, mask = dropout(Tensor(input_x))
    check_dropout_nd_by_keep_prob(func_name, input_x, output.asnumpy(), mask.asnumpy(), keep_prob)
    context.set_context(mode=context.PYNATIVE_MODE)
    dropout.set_inputs(input_dyn)
    output, mask = dropout(Tensor(input_x))
    check_dropout_nd_by_keep_prob(func_name, input_x, output.asnumpy(), mask.asnumpy(), keep_prob)
