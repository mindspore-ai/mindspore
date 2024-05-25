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
import os
import pytest
import mindspore.context as context
from mindspore import Tensor, nn
from mindspore.common import dtype as mstype
from mindspore import Parameter


class GraphNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(10, mstype.float32), name="a")
        self.zero = Parameter(Tensor(0, mstype.float32), name="zero")

    def construct(self, x):
        out = self.zero
        out1 = self.param_a
        if x > 0:
            out = out + self.param_a
        if x > 2:
            out1 += self.param_a
            out += self.param_a
        out1 += self.param_a
        return out, out1

def aoe_online():
    context.set_context(mode=context.GRAPH_MODE, aoe_tune_mode="online", aoe_config={"job_type": "2"})
    context.set_context(jit_config={"jit_level": "O2"})
    net = GraphNet()
    x = Tensor(3, mstype.int32)
    out0, out1 = net(x)
    assert out0 == Tensor(20, mstype.float32)
    assert out1 == Tensor(30, mstype.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_aoe():
    """
    Feature: aoe
    Description: aoe with ge backend.
    Expectation: success.
    """
    aoe_online()
    ret = os.system(f"ls aoe_result_opat_*.json")
    assert ret == 0
