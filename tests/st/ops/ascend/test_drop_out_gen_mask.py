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
import random
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import set_seed
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mask = P.DropoutGenMask(10, 28)
        self.shape = P.Shape()

    def construct(self, x_, y_):
        shape_x = self.shape(x_)
        return self.mask(shape_x, y_)


y = np.array([1.0]).astype(np.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net():
    x = np.ones([2, 4, 2, 2]).astype(np.int32)
    tx = Tensor(x)
    ty = Tensor(1, dtype=mstype.float32)
    mask = Net()
    output = mask(tx, ty)
    print(output.asnumpy())
    assert ([255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] == output.asnumpy()).all()


class Drop(nn.Cell):
    def __init__(self):
        super(Drop, self).__init__()
        self.drop = nn.Dropout(p=0.5)

    def construct(self, out):
        out = self.drop(out)
        return out


def train(net, data):
    net.set_train(True)
    res_list = []
    for _ in range(5):
        res = net(data)
        res_list.append(res.asnumpy())
    return res_list


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_drop():
    """
    Feature: test dropout gen mask diff in diff step.
    Description: dropout gen mask.
    Expectation: No exception.
    """
    set_seed(1)
    np.random.seed(1)
    random.seed(1)
    data = Tensor(np.ones([1, 50]).astype(np.float32))

    net = Drop()
    out_list = train(net, data)

    for i in range(len(out_list)):
        for j in range(len(out_list)):
            if i == j:
                continue
            assert np.allclose(out_list[i], out_list[j], 0, 0) is False


class Net0(nn.Cell):
    def __init__(self):
        super(Net0, self).__init__()
        self.mask_1 = P.DropoutGenMask(10, 10)
        self.mask_2 = P.DropoutGenMask(10, 10)
        self.shape = P.Shape()

    def construct(self, x_, y_):
        shape_x = self.shape(x_)
        out_1 = self.mask_1(shape_x, y_)
        out_2 = self.mask_2(shape_x, y_)
        return out_1, out_2


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.mask_1 = P.DropoutGenMask(20, 20)
        self.mask_2 = P.DropoutGenMask(10, 10)
        self.shape = P.Shape()

    def construct(self, x_, y_):
        shape_x = self.shape(x_)
        out_1 = self.mask_1(shape_x, y_)
        out_2 = self.mask_2(shape_x, y_)
        return out_1, out_2


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.mask_1 = P.DropoutGenMask(10, 10)
        self.mask_2 = P.DropoutGenMask(20, 20)
        self.shape = P.Shape()

    def construct(self, x_, y_):
        shape_x = self.shape(x_)
        out_1 = self.mask_1(shape_x, y_)
        out_2 = self.mask_2(shape_x, y_)
        return out_1, out_2


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_diff_seed():
    """
    Feature: test dropout gen mask diff by diff seed.
    Description: dropout gen mask.
    Expectation: No exception.
    """
    net_0 = Net0()
    net_1 = Net1()
    net_2 = Net2()

    px = np.ones([2, 4, 2, 2]).astype(np.int32)
    py = Tensor(0.5, dtype=mstype.float32)
    net0_out0, net0_out1 = net_0(Tensor(px), py)
    net1_out0, net1_out1 = net_1(Tensor(px), py)
    net2_out0, net2_out1 = net_2(Tensor(px), py)

    assert (np.allclose(net0_out0.asnumpy(), net1_out0.asnumpy(), 0, 0) is False) or \
           (np.allclose(net0_out1.asnumpy(), net1_out1.asnumpy(), 0, 0) is False)
    assert (np.allclose(net0_out0.asnumpy(), net2_out0.asnumpy(), 0, 0) is False) or \
           (np.allclose(net0_out1.asnumpy(), net2_out1.asnumpy(), 0, 0) is False)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fuzz():
    """
    Feature: test dropout gen mask fuzz input.
    Description: dropout gen mask.
    Expectation: ValueError.
    """
    test_op = ops.DropoutGenMask()
    with pytest.raises(ValueError):
        output = test_op(**{})
        print(output)
