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
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.upsample = P.ResizeNearestNeighbor((2, 2))

    @jit
    def construct(self, images):
        return self.upsample(images)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, images, grads):
        return self.grad(self.network)(images, grads)


def test_net():
    image = np.random.random(size=(32, 3, 16, 16)).astype(np.float32)
    grads = np.random.random(size=(32, 3, 2, 2)).astype(np.float32)
    grad = Grad(Net())
    output = grad(Tensor(image), Tensor(grads))
    print("=================output====================")
    print(output)
