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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

class NetNorm(nn.Cell):
    def __init__(self):
        super(NetNorm, self).__init__()

        self.norm_1 = nn.Norm(axis=0)
        self.norm_2 = nn.Norm(axis=1)
        self.norm_3 = nn.Norm(axis=-1)
        self.norm_4 = nn.Norm(axis=-1, keep_dims=True)

    @jit
    def construct(self, indices):
        return (self.norm_1(indices),
                self.norm_2(indices),
                self.norm_3(indices),
                self.norm_4(indices))

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_norm():
    loss = 1e-5
    norm = NetNorm()
    indices = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
    output = norm(indices)
    expect_0 = np.array([4.472136, 4.1231055, 9.486833, 6.0827627]).astype(np.float32)
    expect_1 = np.array([10.677078, 7.071068]).astype(np.float32)
    expect_2 = np.array([10.677078, 7.071068]).astype(np.float32)
    expect_3 = np.array([[10.677078], [7.071068]]).astype(np.float32)

    assert np.allclose(output[0].asnumpy(), expect_0, loss)
    assert np.allclose(output[1].asnumpy(), expect_1, loss)
    assert np.allclose(output[2].asnumpy(), expect_2, loss)
    assert np.allclose(output[3].asnumpy(), expect_3, loss)
