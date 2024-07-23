# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.ops.operations as op
from mindspore import Tensor, nn, ops
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")


# get dynamic shape tensor for input tensor
def dyn_shape(arg):
    return Tensor(shape=[None for _ in arg.shape], dtype=arg.dtype) \
        if isinstance(arg, Tensor) and arg.shape != () else arg


def get_dyn_inputs(input_list):
    return [dyn_shape(arg) for arg in input_list]


class SplitNet(nn.Cell):
    def __init__(self, axis, output_num):
        super().__init__()
        self.split = op.Split(axis=axis, output_num=output_num)
        self.addn = op.AddN()

    def construct(self, inoput_x):
        t = self.split(inoput_x)
        return self.addn(t)


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x):
        grad_fn = self.grad_op(self.net)
        return grad_fn(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_split_grad_dynamic():
    """
    Feature: Test operator Split grad (i.e. Concat) with dynamic input in PyNative mode
    Description:  Test Split grad when call ACL operator
    Expectation: Grad of Split can run and the result is correct.
    """
    input_np = np.ones([2, 1, 8, 1], dtype=np.float32) * 2
    input_x = Tensor(input_np)
    net = SplitNet(2, 2)
    grad_net = GradNetWrtX(net)
    inputs = [input_x]
    grad_net.set_inputs(*get_dyn_inputs(inputs))
    out = grad_net(input_x)
    expected = np.ones([2, 1, 8, 1], dtype=np.float32)
    assert np.allclose(expected, out.asnumpy())
