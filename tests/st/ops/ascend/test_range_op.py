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

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class RangeNet(nn.Cell):
    def __init__(self, maxlen=50):
        super(RangeNet, self).__init__()
        self.range = P.Range(maxlen)

    def construct(self, start, limit, delta):
        return self.range(start, limit, delta)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_range_int64():
    """
    Feature: test Range op on Ascend.
    Description: test the Range when input is int64.
    Expectation: result is right.
    """
    range_net = RangeNet()
    ms_out = range_net(Tensor(2, mstype.int64), Tensor(5, mstype.int64), Tensor(1, mstype.int64)).asnumpy()
    np_expected = np.array([2, 3, 4])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-24, mstype.int64), Tensor(1, mstype.int64), Tensor(4, mstype.int64)).asnumpy()
    np_expected = np.array([-24, -20, -16, -12, -8, -4, 0])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8, mstype.int64), Tensor(1, mstype.int64), Tensor(-1, mstype.int64)).asnumpy()
    np_expected = np.array([8, 7, 6, 5, 4, 3, 2])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(3, mstype.int64), Tensor(-11, mstype.int64), Tensor(-5, mstype.int64)).asnumpy()
    np_expected = np.array([3, -2, -7])
    np.testing.assert_array_equal(ms_out, np_expected)
