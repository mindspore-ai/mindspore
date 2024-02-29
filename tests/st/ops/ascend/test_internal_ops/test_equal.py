# Copyright 2024 Huawei Technologies Co., Ltd
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

import random
import os
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Profiler
import mindspore.common.dtype as mstype

class EqualNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.equal = ops.Equal()

    def construct(self, input_x, input_y):
        out = self.equal(input_x, input_y)
        return out


def test_equal_net():
    """
    Feature: equal test case
    Description: equal test case
    Expectation: the result is correct
    """
    net = EqualNet()

    in1 = [1,2,3,4,5,6,5,4,3,2]
    in2 = [1,3,4,5,5,8,9,4,3,2]
    out = np.array(in1) == np.array(in2)

    fp32_in1 = Tensor(in1, mstype.float32)
    fp32_in2 = Tensor(in2, mstype.float32)
    fp32_out = net(fp32_in1, fp32_in2)
    assert fp32_out.dtype == mstype.bool_
    assert np.allclose(fp32_out.asnumpy(), out, 0.01, 0.01)
    print("fp32 equal success.")


    fp16_in1 = Tensor(in1, mstype.float16)
    fp16_in2 = Tensor(in2, mstype.float16)
    fp16_out = net(fp16_in1, fp16_in2)
    assert fp16_out.dtype == mstype.bool_
    assert np.allclose(fp16_out.asnumpy(), out, 0.01, 0.01)
    print("fp16 equal success.")
