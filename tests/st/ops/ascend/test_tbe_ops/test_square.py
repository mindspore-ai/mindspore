# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.square = P.Square()

    def construct(self, x):
        return self.square(x)


arr_x = np.array([1.0, 4.0, 9.0]).astype(np.float32)


def test_net():
    """
    Feature: test square function.
    Description: test square op forward.
    Expectation: expect correct result.
    """
    square = Net()
    output = square(Tensor(arr_x))
    print(arr_x)
    print(output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_bf16():
    """
    Feature: test square function.
    Description: test square op forward bf16.
    Expectation: expect correct result.
    """
    square = Net()
    output = square(Tensor(arr_x, mindspore.bfloat16))
    except_out = np.array([1., 16., 81.]).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), except_out, 0.004, 0.004)
