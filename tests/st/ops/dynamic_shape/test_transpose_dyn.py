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
import mindspore.ops as ops
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, perm_in):
        super(Net, self).__init__()
        self.transpose = ops.Transpose()
        self.perm = perm_in

    def construct(self, input_):
        x = self.transpose(input_, self.perm)
        return x


def dyn_case():
    perm = (1, 0, 2)
    in_shape = (2, 4, 8)
    np_value = np.random.uniform(0, 20, size=in_shape).astype(np.float16)
    real_input = Tensor(np_value)

    # dynamic transpose
    dyn_transpose = Net(perm)
    dyn_input = Tensor(shape=[None for _ in real_input.shape], dtype=real_input.dtype)
    dyn_transpose.set_inputs(dyn_input)
    dyn_out = dyn_transpose(real_input)

    # static transpose
    static_transpose = Net(perm)
    static_out = static_transpose(real_input)

    np.allclose(dyn_out.asnumpy(), static_out.asnumpy(), 1e-6, 1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_transpose_dyn_cpu(mode):
    """
    Feature: test Transpose dynamic shape on CPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=mode, device_target="CPU")
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_transpose_dyn_gpu(mode):
    """
    Feature: test Transpose dynamic shape on GPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=mode, device_target="GPU")
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_transpose_dyn_ascend(mode):
    """
    Feature: test Transpose dynamic shape on Ascend.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=mode, device_target="Ascend")
    dyn_case()
