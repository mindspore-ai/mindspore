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
""" test graph clear statement. """
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensorarray_clear():
    """
    Feature: Support clear is isolated node.
    Description: Support clear is isolated node.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, dtype, element_shape):
            super(Net, self).__init__()
            self.ta = nn.TensorArray(dtype=dtype, element_shape=element_shape)
            self.index_1 = 1
            self.index_2 = 30

        def construct(self, input_1, input_2):
            size_1 = self.ta.size()
            self.ta.write(self.index_1, input_1)
            self.ta.write(self.index_2, input_2)
            size_2 = self.ta.size()
            self.ta.clear()
            size_3 = self.ta.size()
            return size_1, size_2, size_3

    input_np_1 = np.random.randn(2, 3, 4, 5, 6).astype(np.int32)
    input_np_2 = np.random.randn(2, 3, 4, 5, 6).astype(np.int32)
    net = Net(dtype=ms.int32, element_shape=(2, 3, 4, 5, 6))
    out_ms = net(Tensor(input_np_1), Tensor(input_np_2))
    assert out_ms[0] == 0
    assert out_ms[1] == 31
    assert out_ms[2] == 0
