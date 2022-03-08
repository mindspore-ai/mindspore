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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self, axis=0):
        super(Net, self).__init__()
        self.unique = P.Unique()
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=axis)

    def construct(self, x1, x2):
        out1_unique, _ = self.unique(x1)
        out2_unique, _ = self.unique(x2)
        out1_shape = self.reshape(out1_unique, (1, -1, 2))
        out2_shape = self.reshape(out2_unique, (1, -1, 2))
        return self.concat((out1_shape, out2_shape))

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_concat():
    x1 = Tensor(np.array([1, 2, 3, 1, 4, 2]), mstype.int32)
    x2 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.int32)
    net = Net(axis=1)
    output = net(x1, x2)
    expect = np.array([[[1, 2], [3, 4], [1, 2], [3, 4], [5, 6]]])
    assert (output.asnumpy() == expect).all()
