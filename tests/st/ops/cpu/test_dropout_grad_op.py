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
""" test_dropout """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.ops.operations import _grad_ops as P


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, keep_prob=0.5):
        super(Net, self).__init__()
        self.dropout_grad = P.DropoutGrad(keep_prob)

    def construct(self, output, mask):
        return self.dropout_grad(output, mask)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dropout_grad_001():
    in_tensor = Tensor(np.array([[[3., 1., 2.]], \
                                 [[4., 1., 4.]]]), mstype.float32)
    in_mask = Tensor(np.array([[[1., 0, 0]], [[1., 1., 0]]]), mstype.float32)
    dropout_grad = Net()
    output = dropout_grad(in_tensor, in_mask)
    print("output:\n", output)

    expect = np.array([[[6., 0., 0.]], [[8., 2., 0.]]]).astype(np.float32)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dropout_grad_002():
    in_tensor = Tensor(np.array([[[3., 1., 2.]], [[4., 1., 4.]]]), mstype.float16)
    in_mask = Tensor(np.array([[[1., 0, 0]], [[1., 1., 0]]]), mstype.float16)
    dropout_grad = Net()
    output = dropout_grad(in_tensor, in_mask)
    print("output:\n", output)

    expect = np.array([[[6., 0., 0.]], [[8., 2., 0.]]]).astype(np.float16)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dropout_grad_003():
    in_tensor = Tensor(np.array([[[3., 1., 2.], [3., 1., 2.]], \
                                 [[4., 1., 4.], [4., 1., 4.]]]), mstype.float16)
    in_mask = Tensor(np.array([[[1., 0, 0], [1., 0, 0]], \
                               [[1., 1., 0], [1., 1., 0]]]), mstype.float16)
    dropout_grad = Net()
    output = dropout_grad(in_tensor, in_mask)
    print("output:\n", output)

    expect = np.array([[[6., 0., 0.], [6., 0., 0.]], \
                       [[8., 2., 0.], [8., 2., 0.]]]).astype(np.float16)
    error = np.ones(shape=[2, 2, 3]) * 1.0e-6

    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dropout_grad_004():
    in_tensor = Tensor(np.array([[6.]]), mstype.float32)
    in_mask = Tensor(np.array([[1.]]), mstype.float32)
    dropout_grad = Net(1.)
    output = dropout_grad(in_tensor, in_mask)
    print("output:\n", output)

    expect = np.array([[6.]]).astype(np.float32)
    error = np.ones(shape=[1]) * 1.0e-6

    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)


@pytest.mark.skip(reason='0 in shape is not support')
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dropout_grad_005():
    in_tensor = Tensor(np.array([[]]), mstype.float32)
    in_mask = Tensor(np.array([[]]), mstype.float32)
    dropout_grad = Net(1.)
    output = dropout_grad(in_tensor, in_mask)
    print("output:\n", output)

    expect = np.array([[]]).astype(np.float32)
    error = np.ones(shape=[]) * 1.0e-6

    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dropout_grad_006():
    in_tensor = Tensor(np.array([[[3., 1., 2.]], [[4., 1., 4.]]]), mstype.float16)
    in_mask = Tensor(np.array([[[1., 0, 0]], [[0., 0., 1.]]]), mstype.float16)
    dropout_grad = Net(0.3333333333)
    output = dropout_grad(in_tensor, in_mask)
    print("output:\n", output)

    expect = np.array([[[9., 0., 0.]], [[0., 0., 12.]]]).astype(np.float16)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)
