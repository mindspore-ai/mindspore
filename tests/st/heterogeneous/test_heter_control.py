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
import pytest
import mindspore as ms
from mindspore import Tensor, nn, dtype
from mindspore.ops import operations as P
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


class NetHeterogeneous(nn.Cell):
    def __init__(self, loop_count=3):
        super().__init__()
        self.loop_count = loop_count
        self.add = P.Add().add_prim_attr("primitive_target", "CPU")
        self.mul = P.Mul().add_prim_attr("primitive_target", "Ascend")

    def construct(self, x):
        num = self.loop_count
        if x > num:
            res = self.add(x, x)
        elif x == num:
            res = self.mul(x, num)
        else:
            res = self.mul(x, x)
        return res


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_heterogeneous(mode):
    """
    Feature: Ascend heterogeneous test.
    Description: Test net run on heterogeneous device CPU and Ascend.
    Expectation: No exception and result is correct.
    """
    ms.set_context(device_target="CPU")
    ms.set_context(mode=mode)
    net = NetHeterogeneous()

    x1 = Tensor(4, dtype.int32)
    out1 = net(x1)
    assert out1.asnumpy() == 8

    x2 = Tensor(3, dtype.int32)
    out2 = net(x2)
    assert out2.asnumpy() == 9

    x3 = Tensor(2, dtype.int32)
    out3 = net(x3)
    assert out3.asnumpy() == 4
