# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P

class Net(Cell):
    def __init__(self, axis=0, epsilon=1e-4):
        super(Net, self).__init__()
        self.norm = P.L2Normalize(axis=axis, epsilon=epsilon)

    def construct(self, x):
        return self.norm(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_l2normalize():
    x = np.random.randint(1, 10, (2, 3, 4, 4)).astype(np.float32)
    expect = x / np.sqrt(np.sum(x**2, axis=0, keepdims=True))
    x = Tensor(x)
    error = np.ones(shape=[2, 3, 4, 4]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    norm_op = Net(axis=0)
    output = norm_op(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)
