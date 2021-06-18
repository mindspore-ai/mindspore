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
"""Generate the mindir for bprop"""
import os
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops._grad as g

context.set_context(mode=context.GRAPH_MODE)
os.environ['GENERATE_MINDIR'] = '1'


class NetRelu(nn.Cell):
    def __init__(self):
        super(NetRelu, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


class NetIdentity(nn.Cell):
    def __init__(self):
        super(NetIdentity, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        return self.identity(x)


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


def test_relu():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    relu = NetRelu()
    grad = GradNet(relu)
    grad(x)


def test_identity():
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.int64))
    identity = NetIdentity()
    grad = GradNet(identity)
    grad(x)


test_relu()
test_identity()
# mindspore/ops/_grad/__init__.py
bprop_path = g.__file__
bprop_mindir_path = bprop_path[: bprop_path.rindex('/')] + "/../bprop_mindir/"
print("The new bprop mindir files has been generated in the path \"" + bprop_mindir_path +
      "\", copy the *.mindir to your PYTHONPATH if necessary.")
