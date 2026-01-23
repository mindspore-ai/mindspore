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

"""test aoe."""

import os
from mindspore import context
import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common import dtype as mstype
from mindspore import Parameter
from tests.device_utils import set_device
from tests.mark_utils import arg_mark


class GraphNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(10, mstype.float32), name="a")
        self.zero = Parameter(Tensor(0, mstype.float32), name="zero")

    def construct(self):
        out = self.zero
        out1 = self.param_a

        out = out + self.param_a
        out1 += self.param_a
        out += self.param_a
        out1 += self.param_a
        return out, out1


def aoe_online():
    context.set_context(mode=context.GRAPH_MODE)
    set_device()
    ms.device_context.ascend.op_tuning.aoe_tune_mode("online")
    ms.device_context.ascend.op_tuning.aoe_job_type("2")
    context.set_context(jit_config={"jit_level": "O2"})
    net = GraphNet()
    out0, out1 = net()
    assert out0 == Tensor(30, mstype.float32)
    assert out1 == Tensor(40, mstype.float32)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
def test_aoe():
    """
    Feature: aoe
    Description: aoe with ge backend.
    Expectation: success.
    """
    aoe_online()
    ret = os.system("ls aoe_result_opat_*.json")
    assert ret == 0
