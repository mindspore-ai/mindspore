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
import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import Identity
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.identity = Identity()

    def construct(self, x):
        return self.identity(x)


def generate_testcases(nptype, ms_type=None):
    x = np.random.randn(3, 4, 5, 6).astype(nptype)
    net = Net()
    input_tensor = Tensor(x, ms_type)
    output = net(input_tensor)
    if ms_type == ms.bfloat16:
        np.testing.assert_almost_equal(input_tensor.float().asnumpy(), output.float().asnumpy())
    else:
        np.testing.assert_almost_equal(input_tensor.asnumpy(), output.asnumpy())
    assert id(input_tensor) != id(output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_identity_bfloat16(mode):
    """
    Feature: test Identity forward.
    Description: test bfloat16 inputs.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    generate_testcases(np.float32, ms_type=ms.bfloat16)
