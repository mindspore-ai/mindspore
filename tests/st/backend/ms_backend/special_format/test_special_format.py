# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test special format"""
import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import ops, jit
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    @jit
    def construct(self, x, other):
        return ops.add(x, other)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_special_format_for_input_parameter():
    """
    Feature: special format test case
    Description: test special format in dynamic-shape input and following operator in inference
    Expectation: no exception
    """
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = Net()
    # FRACTAL_NZ
    x = [ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2], dtype=mstype.float32), 29),
         ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2], dtype=mstype.float32), 29)]
    output = net(x[0], x[1])
    print(output)

    # NC1HWC0
    x = [ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 3),
         ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 3)]
    output = net(x[0], x[1])
    print(output)

    # FRACTAL_Z
    x = [ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 4),
         ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 4)]
    output = net(x[0], x[1])
    print(output)

    # NDC1HWC0
    x = [ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 32),
         ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 32)]
    output = net(x[0], x[1])
    print(output)

    # FRACTAL_Z_3D
    x = [ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 33),
         ms.ops.auto_generate.format_cast(ms.mint.zeros([2, 2, 2, 2], dtype=mstype.float32), 33)]
    output = net(x[0], x[1])
    print(output)
