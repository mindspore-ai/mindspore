# Copyright 2021 Huawei Technologies Co., Ltd
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


class EqualCount(nn.Cell):
    def __init__(self):
        super(EqualCount, self).__init__()
        self.op = P.EqualCount()

    def construct(self, *inp):
        return self.op(*inp)

def get_output(*inp, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    output = EqualCount()(*inp)
    return output

def basic_test(datatype):
    x = Tensor(np.array([[1, 1, 1, 1], [3, 3, 3, 3]]).astype(datatype))
    y = Tensor(np.array([[1, 2, 1, 2], [1, 1, 3, 3]]).astype(datatype))
    expect = get_output(x, y, enable_graph_kernel=False)
    output = get_output(x, y, enable_graph_kernel=True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    basic_test(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_fp32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    basic_test(np.float32)
