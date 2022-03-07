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
# ==============================================================================
"""
Watchpoints test script for dump analyze_fail.dat when infer failed.
"""
# pylint: disable=too-many-function-args
import os
import pytest
import mindspore
from mindspore import ops, Tensor, nn
from tests.security_utils import security_off_wrap


@security_off_wrap
def test_infer_fail_generate_analyze_fail_dat():
    """
    Feature: test dump analyze_fail.dat.
    Description: test dump analyze_fail.dat if infer failed.
    Expectation: success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.sub = ops.Sub()
            self.mul = ops.Mul()
            self.div = ops.Div()

        def func(self, x, y):
            return self.div(x, y)

        def construct(self, x, y):
            a = self.sub(x, 1)
            b = self.add(a, y)
            c = self.mul(b, self.func(a, a, b))
            return c

    input1 = Tensor(3, mindspore.float32)
    input2 = Tensor(2, mindspore.float32)
    net = Net()

    with pytest.raises(TypeError) as excinfo:
        net(input1, input2)
    assert "rank_0/om/analyze_fail.dat" in str(excinfo.value)
    assert os.path.exists("./rank_0/om/analyze_fail.dat") is True
