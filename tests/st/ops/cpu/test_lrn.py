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
from mindspore.ops import operations as P


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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
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
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    benchmark_output = lrn_np_bencmark(data_type)
    lrn = LrnNet(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
    output = lrn(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = lrn(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
