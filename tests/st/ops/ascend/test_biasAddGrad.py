# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.operations import _grad_ops as G

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add_grad = G.BiasAddGrad()
        # self.dout = Parameter(initializer(
        # 'normal', [2, 3, 3, 4]), name='dout')

    @jit
    def construct(self, dout_):
        return self.bias_add_grad(dout_)


dout = np.ones([2, 3, 4, 4]).astype(np.float32)
bias_add_grad = Net()
output = bias_add_grad(Tensor(dout))
expect_output = np.array([32., 32., 32.]).astype(np.float32)
assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"
print(output.asnumpy())
