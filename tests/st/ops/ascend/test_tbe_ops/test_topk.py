# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, k):
        super(Net, self).__init__()
        self.topk = P.TopK(True)
        self.k = k

    def construct(self, x):
        return self.topk(x, self.k)


def test_net():
    x = np.random.randn(4, 4).astype(np.float16)
    k = 2
    TopK = Net(k)
    output = TopK(Tensor(x))
    print("***********x*********")
    print(x)

    print("***********output y*********")
    print(output[0].asnumpy())
