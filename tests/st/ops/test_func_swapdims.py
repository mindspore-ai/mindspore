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
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
from mindspore import Tensor


class NetSwapDims(nn.Cell):
    def construct(self, x, dim0, dim1):
        return ops.swapdims(x, dim0=dim0, dim1=dim1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_swapdims(mode):
    """
    Feature: swapdims
    Description: Verify the result of swapdims
    Expectation: success.
    """
    ms.set_context(mode=mode)
    swapdims_op = NetSwapDims()

    np_array = np.random.random((3, 4, 5)).astype('float32')
    x = Tensor(np_array)

    output1 = swapdims_op(x, 0, 2)
    expected1 = np.swapaxes(np_array, 0, 2)
    assert np.allclose(output1.asnumpy(), expected1)

    output2 = swapdims_op(x, 1, 1)
    expected2 = np.swapaxes(np_array, 1, 1)
    assert np.allclose(output2.asnumpy(), expected2)
