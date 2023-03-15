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

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Im2ColNet(nn.Cell):
    def __init__(self, ksizes, strides=1, dilations=1, pads=0):
        super(Im2ColNet, self).__init__()
        self.ksizes = ksizes
        self.strides = strides
        self.dilations = dilations
        self.pads = pads
        self.im2col = P.Im2Col(ksizes=self.ksizes, strides=self.strides, dilations=self.dilations, pads=self.pads)

    def construct(self, x):
        output = self.im2col(x)
        return output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_im2col():
    """
    Feature: Test tbe for im2col operator
    Description: Test im2col with input tensor's type float16, ksizes=3
    Expectation: Consistent with the assertion
    """
    ksizes = 3
    x = Tensor(input_data=np.random.rand(4, 4, 32, 32), dtype=mstype.float16)
    im2col = Im2ColNet(ksizes)
    outputs = im2col(Tensor(x))
    assert outputs.shape == (4, 4, 9, 900)
