# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G


class NetMaskedSelectGrad(nn.Cell):
    def __init__(self):
        super(NetMaskedSelectGrad, self).__init__()
        self.masked_select_grad_fun = G.MaskedSelectGrad()

    def construct(self, x, mask, output_grad):
        return self.masked_select_grad_fun(x, mask, output_grad)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_masked_select_grad():
    """
    Feature: MaskedSelectGrad
    Description:  test cases for MaskedSelectGrad operator.
    Expectation: the result match expect.
    """
    input_tensor = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
    mask = Tensor(np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool))
    output_grad = Tensor(np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]).astype(np.int32))
    expect_result = (np.array([4, 8, 12, 16]))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    masked_select_grad = NetMaskedSelectGrad()
    input_grad = masked_select_grad(input_tensor, mask, output_grad)
    assert (input_grad.asnumpy() == expect_result).all()
