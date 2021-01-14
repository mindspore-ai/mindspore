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
""" test outermost net pass non_tensor inputs"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE)


def test_outermost_net_pass_scalar_tuple_list_dict():
    class TestNet(nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()

        def construct(self, tuple_a, z, list_m, w, s, dict_n):
            return z - tuple_a[2] + list_m[1][1]["x"] - w + s - dict_n["y"]

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.forward_net = net
            self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
            self.grad_all = C.GradOperation(get_all=True)

        def construct(self, tuple_a, z, list_m, w, s, dict_n):
            return self.grad_all(self.forward_net)(tuple_a, z, list_m, w, s, dict_n)

    x = Tensor(np.ones((2, 2), np.float32))
    y = Tensor(np.ones((2, 2), np.float32) * 2)
    z = Tensor(np.ones((2, 2), np.float32) * 3)
    w = Tensor(np.ones((2, 2), np.float32) * 4)
    arg_t0 = (x, y, z, w)
    arg_t1 = (w, y, z, w)
    arg_l0 = [[x, x], [[x, y], {"x": x, "y": y, "z": x, "p": y}]]
    arg_l1 = [[x, x], [[x, y], {"x": x, "y": y, "z": x, "p": y}]]
    args_d0 = {"x": x, "y": y}
    args_d1 = {"x": x, "y": y}
    forward_net = TestNet()
    forward_net(arg_t0, z, arg_l0, w, 6, args_d0)
    forward_net(arg_t1, z, arg_l1, x, 6, args_d1)

    grad_net = GradNet(forward_net)
    grad_net(arg_t0, z, arg_l0, w, 6, args_d0)
    grad_net(arg_t1, z, arg_l1, x, 6, args_d1)
