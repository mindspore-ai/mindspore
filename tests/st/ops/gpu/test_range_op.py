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

class RangeNet(nn.Cell):
    def __init__(self):
        super(RangeNet, self).__init__()
        self.range = P.Range()

    def construct(self, s, e, d):
        return self.range(s, e, d)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_int():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    range_net = RangeNet()
    ms_out = range_net(Tensor(2, mstype.int32), Tensor(5, mstype.int32), Tensor(1, mstype.int32)).asnumpy()
    np_expected = np.array([2, 3, 4])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-24, mstype.int32), Tensor(1, mstype.int32), Tensor(4, mstype.int32)).asnumpy()
    np_expected = np.array([-24, -20, -16, -12, -8, -4, 0])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8, mstype.int32), Tensor(1, mstype.int32), Tensor(-1, mstype.int32)).asnumpy()
    np_expected = np.array([8, 7, 6, 5, 4, 3, 2])
    np.testing.assert_array_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(3, mstype.int32), Tensor(-11, mstype.int32), Tensor(-5, mstype.int32)).asnumpy()
    np_expected = np.array([3, -2, -7])
    np.testing.assert_array_equal(ms_out, np_expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_float():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    range_net = RangeNet()
    ms_out = range_net(Tensor(2.3, mstype.float32), Tensor(5.5, mstype.float32), Tensor(1.2, mstype.float32)).asnumpy()
    np_expected = np.array([2.3, 3.5, 4.7])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(-4, mstype.float32), Tensor(-1, mstype.float32), Tensor(1.5, mstype.float32)).asnumpy()
    np_expected = np.array([-4.0, -2.5])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(8.0, mstype.float32), Tensor(1.0, mstype.float32), Tensor(-1.0, mstype.float32)).asnumpy()
    np_expected = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

    range_net = RangeNet()
    ms_out = range_net(Tensor(1.5, mstype.float32), Tensor(-1, mstype.float32), Tensor(-18.9, mstype.float32)).asnumpy()
    np_expected = np.array([1.5])
    np.testing.assert_array_almost_equal(ms_out, np_expected)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_range_invalid_max_output_length():
    with pytest.raises(ValueError):
        _ = P.Range(0)
        _ = P.Range(-1)
        _ = P.Range(None)
        _ = P.Range('5')
