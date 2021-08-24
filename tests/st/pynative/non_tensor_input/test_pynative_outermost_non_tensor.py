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
""" test outermost net pass non_tensor inputs"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()
        self.sub = P.Sub()

    def construct(self, tensor_x, tuple_a, list_b, tensor_y, tensor_z, dict_c):
        out = self.add(tensor_x, tuple_a[0])
        out = self.sub(out, list_b[1][1]["y"])
        out = self.add(out, tensor_y)
        out = self.sub(out, tensor_z)
        out = self.add(out, dict_c["u"])
        return out


class GradNet(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_x, list_b, tensor_y, tensor_z, dict_c):
        return self.grad_all(self.forward_net)(tuple_a, tensor_x, list_b, tensor_y, tensor_z, dict_c)


x = Tensor(np.ones((2, 2), np.float32))
y = Tensor(np.ones((2, 2), np.float32) * 2)
z = Tensor(np.ones((2, 2), np.float32) * 3)
w = Tensor(np.ones((2, 2), np.float32) * 4)
p = Tensor(np.ones((2, 2), np.float32) * 5)
u = Tensor(np.ones((2, 2), np.float32) * 6)
arg_t0 = (x, y, z, w)
arg_l0 = [[x, x], [[x, y], {"x": x, "y": y, "z": z, "p": p}]]
args_d0 = {"x": x, "y": y, "u": u}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_non_tensor_inputs():
    # grad first input
    grad_fist_input_tensor_net = GradNet(Net(), get_all=False)
    ret = grad_fist_input_tensor_net(z, arg_t0, arg_l0, w, p, args_d0)
    assert np.allclose(ret.asnumpy(), np.ones((2, 2), np.float32))
    # grad all inputs
    grad_all_input_tensor_net = GradNet(Net(), get_all=True)
    ret_all = grad_all_input_tensor_net(z, arg_t0, arg_l0, w, p, args_d0)
    assert len(ret_all) == 3
    assert np.allclose(ret_all[0].asnumpy(), np.ones((2, 2), np.float32))
    assert np.allclose(ret_all[1].asnumpy(), np.ones((2, 2), np.float32))
    assert np.allclose(ret_all[2].asnumpy(), np.ones((2, 2), np.float32) * -1)
