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

import pytest
import numpy as np
import mindspore.context as context
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor


class PadNet(nn.Cell):
    def __init__(self, paddings):
        super(PadNet, self).__init__()
        self.pad = ops.Pad(paddings)

    def construct(self, x):
        return self.pad(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32,
                                   np.int64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_pad_dtype(mode, dtype):
    """
    Feature: test ops.Pad forward.
    Description: inputs with different data type.
    Expectation: the result match with expect
    """
    context.set_context(mode=mode, device_target="GPU")
    paddings = ((2, 1), (3, 1))
    x = np.arange(4 * 6).reshape((4, 6)).astype(dtype)
    expect = np.pad(x, paddings, mode="constant", constant_values=0)
    net = PadNet(paddings)
    output = net(Tensor(x))
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)
