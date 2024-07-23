# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore import Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.gather = ops.gather_elements

    def construct(self, x, dim, index):
        return self.gather(x, dim, index)


def generate_testcases(ms_type=None):
    input_tensor = Tensor([[3, 4, 5],
                           [6, 7, 8],
                           [9, 10, 11]]).astype(ms_type)
    index = Tensor([[2, 1, 0]])
    dim = 0
    net = Net()
    output = net(input_tensor, dim, index)
    except_np = np.array([9., 7., 5.])
    rtol = 1.e-6
    atol = 1.e-6
    assert np.allclose(output.float().asnumpy(), except_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_gather_elements_bfloat16(mode):
    """
    Feature: test gather_elements forward.
    Description: test bfloat16 inputs.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    generate_testcases(ms_type=ms.bfloat16)
