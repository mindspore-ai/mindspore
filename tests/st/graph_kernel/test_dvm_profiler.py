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
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor, nn
from mindspore import Profiler


class Net(nn.Cell):
    def construct(self, x, y):
        a = x + y + 4
        b = x - y - 5
        return a * b


def fuse(shape1, shape2, dtype):
    np.random.seed(1)
    i0 = Tensor(np.random.uniform(1, 2, shape1).astype(dtype))
    i1 = Tensor(np.random.uniform(1, 2, shape2).astype(dtype))
    context.set_context(enable_graph_kernel=True)
    profiler = Profiler()
    net_obj = Net()
    output = net_obj(i0, i1)
    profiler.analyse()
    print(output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dvm_profiling():
    """
    Feature: easy test case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: Timeline generated successfully.
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE)
    fuse((32, 1024), (32, 1024), np.float16)
