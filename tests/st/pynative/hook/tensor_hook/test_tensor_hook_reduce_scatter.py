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
import mindspore.nn as nn
import mindspore as ms
from mindspore.communication.management import init, get_rank
from mindspore.communication.comm_func import reduce_scatter_tensor
from mindspore import Tensor, Parameter
from mindspore import ops
from tests.st.pynative.utils import GradOfAllParams

np.random.seed(1)

init()


def hook_fn(grad_out):
    """改变梯度"""
    print("hook_fn print grad_out:", grad_out, flush=True)  # 该梯度是传播到该tensor时，该tensor所对应的梯度
    output, _ = reduce_scatter_tensor(grad_out, ops.ReduceOp.SUM)
    return output


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.weight1 = Parameter(Tensor(np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), ms.float32), name="weight1")
        self.weight1.register_hook(hook_fn)

    def construct(self, x):
        x = x * self.weight1
        return x


def test_tensor_hook_reduce_scatter_8p():
    """
    Feature: test 'ReduceScatter' communication operator.
    Description: test 'ReduceScatter' communication operator.
    Expectation: expect correct result.
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), ms.float32)
    net1 = Net()
    ms_grad = GradOfAllParams(net1, False)
    output = ms_grad(input_x)
    if get_rank() == 3:
        assert np.allclose(output[0].asnumpy(), Tensor(np.array([32])).astype(np.float32).asnumpy(), 0.001, 0.001)

    if get_rank() == 7:
        assert np.allclose(output[0].asnumpy(), Tensor(np.array([64])).astype(np.float32).asnumpy(), 0.001, 0.001)
