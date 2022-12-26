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
from mindspore.train import Model

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return self.tanh(x)


input_shape = [1]
input_np = np.random.randn(*input_shape).astype(np.float32)
input_me = Tensor(input_np)


def test_net():
    context.set_context(mode=context.GRAPH_MODE)
    tanh = Net()
    tanh.set_train()
    m = Model(tanh)
    out = m.predict(input_me)
    print("out_me.dtype={}".format(out.dtype))
    print("out_me.asnumpy={}".format(out.asnumpy()))
    return out.asnumpy()
