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
from mindspore import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(device_target="Ascend")


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.maxpoolv2 = P.MaxPoolWithArgmaxV2(kernel_size=2, strides=2, pads=0, dilation=1, ceil_mode=False,
                                               argmax_type=mstype.int64)

    @jit
    def construct(self, x):
        output = self.maxpoolv2(x)
        return output[0]


def test_maxpool_grad_with_argmax_v2():
    x = Tensor(np.arange(6 * 6).reshape(1, 1, 6, 6).astype(np.float16))
    index = Tensor(np.array([[[
        [0.7, 0.9, 0.11],
        [0.19, 0.21, 0.23],
        [0.31, 0.33, 0.35]
    ]]]).astype(np.float16))
    net = Grad(Net())
    output = net(x, index)
    print("***********output output*********")
    print(output[0].asnumpy())
