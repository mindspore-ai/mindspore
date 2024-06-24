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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax2d = nn.Softmax2d()

    def construct(self, x):
        return self.softmax2d(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softmax2d_normal():
    """
    Feature: Softmax2d
    Description: Verify the result of Softmax2d
    Expectation: success
    """
    net = Net()
    a = Tensor(np.array([[[[0.1, 0.2]], [[0.3, 0.4]], [[0.6, 0.5]]]]).astype(np.float32))
    output = net(a)
    expected_output = np.array([[[[0.258, 0.28]], [[0.316, 0.342]], [[0.426, 0.378]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expected_output, 1e-3, 1e-3)
