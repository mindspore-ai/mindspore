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
from mindspore import Tensor
from mindspore.ops import functional as F


def test_batch_norm_forward_functional(nptype):
    """
    Feature: test batch_norm forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones([2, 2]).astype(nptype))
    running_mean = Tensor(np.ones([2]).astype(nptype))
    running_var = Tensor(np.ones([2]).astype(nptype))
    weight = Tensor(np.ones([2]).astype(nptype))
    bias = Tensor(np.ones([2]).astype(nptype))
    output = F.batch_norm(input_x, running_mean, running_var, weight, bias)
    expected = np.array([[1., 1.], [1., 1.]]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batch_norm_forward_float32_functional():
    """
    Feature: test batch_norm forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_batch_norm_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_batch_norm_forward_functional(np.float32)
