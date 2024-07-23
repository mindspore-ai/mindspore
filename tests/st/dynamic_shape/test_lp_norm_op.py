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
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.common import dtype as ms_type
from tests.mark_utils import arg_mark


class LpNormNet(nn.Cell):
    def __init__(self, axis, p=2, keep_dims=False, epsilon=1e-12):
        super(LpNormNet, self).__init__()
        self.lp_norm = P.LpNorm(axis, p, keep_dims, epsilon)

    def construct(self, input_x):
        output = self.lp_norm(input_x)
        return output


def lp_norm_np_bencmark(data_type):
    """
    Feature: generate a LpNorm numpy benchmark.
    Description: The input shape need to match input shape.
    Expectation: match to np mindspore LpNorm.
    """
    result = np.array([9.165152, 10.954452]).astype(data_type)
    return result


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_lp_norm_dy_shape(data_type):
    """
    Feature: Test LpNorm DynamicShape.
    Description: The input data type only float16 and float32.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    axis = [0, 1]
    p = 2
    keep_dims = False
    lp_norm_net = LpNormNet(axis, p, keep_dims)
    input_x_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(data_type)
    loss = 1e-4
    ms_data_type = ms_type.float32
    if data_type == np.float16:
        ms_data_type = ms_type.float16
        loss = 1e-3
    benchmark_output = lp_norm_np_bencmark(data_type)
    input_dyn = Tensor(shape=[2, 2, None], dtype=ms_data_type)
    lp_norm_net.set_inputs(input_dyn)
    ms_result = lp_norm_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    ms_result = lp_norm_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
