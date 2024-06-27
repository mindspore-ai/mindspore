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
from tests.mark_utils import arg_mark
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, perm):
        super(Net, self).__init__()
        self.transpose = P.Transpose()
        self.perm = perm

    def construct(self, x):
        return self.transpose(x, self.perm)


def get_output(x, perm, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net(perm)
    output = net(x)
    return output


def compare_transpose_result(shape, dtype, perm):
    x = Tensor(np.random.random(shape).astype(dtype))
    expect = get_output(x, perm, False)
    output = get_output(x, perm, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_transpose_to_reshape():
    """
    Feature: Test transpose replacement in arithmetic_simplify pass.
    Description: Verify the correctness of the replacement.
    Expectation: No exception
    """
    context.set_context(mode=context.GRAPH_MODE)
    compare_transpose_result((10, 1, 20, 1, 4, 5), np.float32, (1, 3, 0, 2, 4, 5))
    compare_transpose_result((10, 1, 20, 1, 4, 5), np.float32, (0, 1, 3, 2, 4, 5))
    compare_transpose_result((10, 1, 20, 1, 4, 5), np.float32, (3, 0, 2, 4, 1, 5))
