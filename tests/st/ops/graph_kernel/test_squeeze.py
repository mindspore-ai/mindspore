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


class Net(nn.Cell):
    def __init__(self, axis):
        super(Net, self).__init__()
        self.squeeze = P.Squeeze(axis)

    def construct(self, x):
        return self.squeeze(x)


def get_output(x, axis=(), enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net(axis)
    output = net(x)
    return output


def test_squeeze(shape, dtype, axis=()):
    x = Tensor(np.random.normal(0, 10, shape).astype(dtype))
    expect = get_output(x, axis, False)
    output = get_output(x, axis, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_squeeze_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_squeeze((1, 16, 1, 1), np.int32)
    test_squeeze((1, 16, 1, 1), np.float32, (0, 2))
