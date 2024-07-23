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
import pytest


import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops.operations import _inner_ops as P
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_came_part3 = P.ApplyCamePart3()


    def construct(self, *inputs):
        return self.apply_came_part3(*inputs)

@pytest.mark.skip(reason="only for testing stuck scenario")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net():
    """
    Feature: test apply_came_part3 tensor api.
    Description: test inputs given their dtype.
    Expectation: execute without error.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    apply_came_part3 = Net()
    u = Tensor(np.ones([1024, 64]), dtype=ms.float32)
    m = Tensor(np.ones([1024, 64]), dtype=ms.float32)
    eps = 0.8
    beta1 = 0.5
    clip_threshold = 0.5
    sum_square_u = Tensor(np.array([128]), dtype=ms.float32)
    global_shape = (1024, 64)
    use_first_moment = False
    inputs = [u, m, eps, beta1, clip_threshold, sum_square_u, global_shape, use_first_moment]
    output = apply_came_part3(*inputs)
    print(output[0].float().asnumpy())
