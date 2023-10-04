# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore import Tensor, context
from mindspore.ops import auto_generate as P
import mindspore.common.dtype as mstype
import test_utils


@test_utils.run_with_cell
def argmax_forward_func(x, axis, out_type):
    return P.Argmax(axis, out_type)(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax(mode):
    """
    Feature: Test argmax op.
    Description: Test argmax.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    x = Tensor([[1, 20, 5], [67, 8, 9], [130, 24, 15]], mstype.float32)
    output = argmax_forward_func(x, -1, mstype.int32)
    expect_output = np.array([1, 0, 0]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)
