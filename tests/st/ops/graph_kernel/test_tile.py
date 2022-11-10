# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, multiples):
        super(Net, self).__init__()
        self.tile = P.Tile()
        self.multiples = multiples

    def construct(self, x):
        return self.tile(x, self.multiples)


def get_output(x, multiples, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net(multiples)
    output = net(x)
    return output


def test_tile(shape, dtype, multiples):
    x = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    expect = get_output(x, multiples, False)
    output = get_output(x, multiples, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tile_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_tile((24, 1), np.float16, (2, 2, 2))
    test_tile((24, 1), np.float32, (1, 2))
