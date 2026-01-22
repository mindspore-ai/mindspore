# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
""" test_bprop """

import torch
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, nn, mint, ops, _no_grad
from mindspore.graph.api import _pynative_executor
from mindspore.ops.auto_generate.gen_ops_def import as_strided, transpose, broadcast_to
from tests.st.pynative.utils import GradOfAllInputs, GradOfFirstInput
from tests.mark_utils import arg_mark


class ViewCopyNet(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        view = as_strided(z, (2, 2), (3, 1))
        view.copy_(y)
        return view


def view_copy_backward(x, y):
    x = x * 2
    view = x.as_strided((2, 2), (3, 1))
    view.copy_(y)
    view.sum().backward()


class TensorSliceCopyNet(nn.Cell):
    def construct(self, x):
        z = x * 2
        z[1] = 3
        z = z * 2
        z[2:3] = 3
        z = z * 2
        return z


def tensor_slice_copy_net(x):
    z = x * 2
    z[1] = 3
    z = z * 2
    z[2:3] = 3
    z = z * 2
    z.sum().backward()


class TensorIndexCopyNet(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z[None] = 2
        z = z * y
        z[...] = 1
        z = z * y
        return z


def tensor_index_copy_net(x, y):
    z = x * 2
    z[None] = 2
    z = z * y
    z[...] = 1
    z = z * y
    z.sum().backward()


class CommonViewOpCopyNet1(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z = transpose(z, (1, 0))
        z.copy_(y)
        z = x * y
        return z


def common_view_op_copy_net1(x, y):
    z = x * 2
    z = z.transpose(1, 0)
    z.copy_(y)
    z = x * y
    z.sum().backward()


class CommonViewOpCopyNet2(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z = z.view((3, 2))
        z.copy_(y)
        z = ops.matmul(x, y)
        return z


def common_view_op_copy_net2(x, y):
    z = x * 2
    z = z.view(3, 2)
    z.copy_(y)
    z = torch.matmul(x, y)
    z.sum().backward()


class CommonViewOpCopyNet3(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z = broadcast_to(z, (3, 3))
        z.copy_(y)
        z = x * y
        return z


def common_view_op_copy_net3(x, y):
    z = x * 2
    z = z.broadcast_to(3, 3)
    z.copy_(y)
    z = x * y
    z.sum().backward()


class MultiViewCopyNet(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        view1 = as_strided(z, (2, 2), (3, 1))
        view2 = as_strided(view1, (2, 1), (2, 1))
        view3 = transpose(view2, (1, 0))
        view3.copy_(y)
        return view2


def multi_view_copy_backward(x, y):
    z = x * 2
    view1 = z.as_strided((2, 2), (3, 1))
    view2 = view1.as_strided((2, 1), (2, 1))
    view3 = view2.transpose(1, 0)
    view3.copy_(y)
    view2.sum().backward()


class MultiViewCopyNet2(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z[1] = 1
        view1 = as_strided(z, (2, 2), (3, 1))
        view2 = as_strided(view1, (2, 1), (2, 1))
        view2.copy_(y)
        return z


def multi_view_copy_backward2(x, y):
    z = x * 2
    z[1] = 1
    view1 = z.as_strided((2, 2), (3, 1))
    view2 = view1.as_strided((2, 1), (2, 1))
    view2.copy_(y)
    z.sum().backward()


class MultiViewCopyNet3(nn.Cell):
    def construct(self, x, y, value):
        z = x * y
        view1 = z[2:4, 2:4]
        view2 = z[:2, :2]
        view3 = z[1:3, 1:3]
        view1.copy_(value)
        view2.copy_(value)
        view3.copy_(value)
        res = view1 + view2 + view3
        return res


def multi_view_copy_backward3(x, y, value):
    z = x * y
    view1 = z[2:4, 2:4]
    view2 = z[:2, :2]
    view3 = z[1:3, 1:3]
    view1.copy_(value)
    view2.copy_(value)
    view3.copy_(value)
    res = view1 + view2 + view3
    res.sum().backward()


class MultiCopyViewNet(nn.Cell):
    def construct(self, x, y, z):
        x = x * 2
        view1 = as_strided(x, (2, 2), (3, 1))
        view1.copy_(y)
        view2 = as_strided(view1, (2, 1), (2, 1))
        view2.copy_(z)
        return x


def multi_copy_view_backward(x, y, z):
    x = x * 2
    view1 = x.as_strided((2, 2), (3, 1))
    view1.copy_(y)
    view2 = view1.as_strided((2, 1), (2, 1))
    view2.copy_(z)
    x.sum().backward()


class ConstantTensorCopyViewNet(nn.Cell):
    def construct(self, x):
        x = x * 2
        y = Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        view = y[1]
        view.copy_(x)
        return view


def constant_tensor_copy_view_net(x):
    x = x * 2
    y = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    view = y[1]
    view.copy_(x)
    view.sum().backward()


class AsStridedExpandNet(nn.Cell):
    def construct(self, x):
        y = as_strided(x, (3, 3), (1, 0))
        return y


def as_strided_expand(x):
    y = x.as_strided((3, 3), (1, 0))
    y.sum().backward()


class AsStridedOverlapNet(nn.Cell):
    def construct(self, x):
        y = as_strided(x, (3, 2), (3, 6))
        return y


def as_strided_overlap(x):
    y = x.as_strided((3, 2), (3, 6))
    y.sum().backward()


class AsStridedInputOverlapNet(nn.Cell):
    def construct(self, x):
        y = mint.broadcast_to(x, (3, 3))
        z = as_strided(y, (1, 1), (1, 1))
        return z


def as_strided_input_overlap(x):
    y = x.expand(3, 3)
    z = y.as_strided((1, 1), (1, 1))
    z.sum().backward()


class NoGradViewCopyNet(nn.Cell):
    def construct(self, x, y):
        with _no_grad():
            view = as_strided(x, (2, 2), (2, 1))
        view.copy_(y)
        return view


class MultiOutputViewCopyNet(nn.Cell):
    def construct(self, x, y):
        view = mint.split(x, 2)
        view[0].copy_(y)
        return view


class MultiOutputViewCopyNet2(nn.Cell):
    def construct(self, x, y):
        view1 = mint.split(x, 2)
        view2 = view1[0]
        view2.copy_(y)
        return view2


class LeafViewCopyNet(nn.Cell):
    def construct(self, x, y):
        x.copy_(y)
        return x


class OverlapViewCopyNet(nn.Cell):
    def construct(self, x, y):
        x.copy_(y)
        return x


class MsViewInplaceHasBprop(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        b = as_strided(z, (3, 3), (3, 1))
        b.copy_(y)
        return b

    def bprop(self, x, y, out, dout):
        a = Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        b = Tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        return a, b


class TorchViewInplaceHasBprop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        z = x * 2
        b = z.as_strided((3, 3), (3, 1))
        b.copy_(y)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        a = torch.tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]).astype(np.float32))
        b = torch.tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]).astype(np.float32))
        return a, b


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    view_copy_net = ViewCopyNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=False)
    grads1 = grad_fn1(x1, y1)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    view_copy_backward(x_torch1, y_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y2 = Tensor([[1.0, 1.0]])
    sens = Tensor([[1.0], [1.0]])
    multi_view_net = MultiViewCopyNet()
    multi_view_net.set_inputs()
    grad_fn2 = GradOfAllInputs(multi_view_net, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([[1.0, 1.0]], requires_grad=True)
    multi_view_copy_backward(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_slice_copy():
    """
    Feature: Test tensor slice inplace grad.
    Description: test tensor copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    tensor_slice_copy = TensorSliceCopyNet()
    tensor_slice_copy.set_inputs()
    grad_fn1 = GradOfAllInputs(tensor_slice_copy, sens_param=True)
    grads1 = grad_fn1(x1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], requires_grad=True)
    tensor_slice_copy_net(x_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    y2 = Tensor([[1., 3., 1.], [1.0, 1.0, 3.0], [2.0, 2.0, 3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    tensor_index_copy = TensorIndexCopyNet()
    tensor_index_copy.set_inputs()
    grad_fn2 = GradOfAllInputs(tensor_index_copy, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([[1., 3., 1.], [1.0, 1.0, 3.0], [2.0, 2.0, 3.0]], requires_grad=True)
    tensor_index_copy_net(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_common_view_copy():
    """
    Feature: Test common view inplace grad.
    Description: test tensor copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    y1 = Tensor([[1., 1., 3.], [2.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    net1 = CommonViewOpCopyNet1()
    net1.set_inputs()
    grad_fn1 = GradOfAllInputs(net1, sens_param=True)
    grads1 = grad_fn1(x1, y1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1., 1., 3.], [2.0, 1.0, 1.0], [1.0, 2.0, 3.0]], requires_grad=True)
    common_view_op_copy_net1(x_torch1, y_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y2 = Tensor([[1., 1.], [2.0, 1.0], [1.0, 1.0]])
    sens = Tensor([[1., 1.], [1.0, 1.0]])
    net2 = CommonViewOpCopyNet2()
    net2.set_inputs()
    grad_fn2 = GradOfAllInputs(net2, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([[1., 1.], [2.0, 1.0], [1.0, 1.0]], requires_grad=True)
    common_view_op_copy_net2(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())

    # x3 = Tensor([1., 2., 3.])
    # y3 = Tensor([[1., 1., 2.], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    # sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1., 1., 1.]])
    # net3 = CommonViewOpCopyNet3()
    # net3.set_inputs()
    # grad_fn2 = GradOfAllInputs(net3, sens_param=True)
    # grads3 = grad_fn2(x3, y3, sens)
    # x_torch3 = torch.tensor([1., 2., 3.], requires_grad=True)
    # y_torch3 = torch.tensor([[1., 1., 2.], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]], requires_grad=True)
    # common_view_op_copy_net3(x_torch3, y_torch3)
    # np.testing.assert_almost_equal(grads3[0].asnumpy(), x_torch3.grad.numpy())
    # np.testing.assert_almost_equal(grads3[1].asnumpy(), y_torch3.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_multi_view_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    sens = Tensor([[1.0, 1.0], [1.0, 1.0]])
    view_copy_net = ViewCopyNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=True)
    grads1 = grad_fn1(x1, y1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    view_copy_backward(x_torch1, y_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y2 = Tensor([1.0, 1.0])
    sens = Tensor([[1.0], [1.0]])
    multi_view_net = MultiViewCopyNet()
    multi_view_net.set_inputs()
    grad_fn2 = GradOfAllInputs(multi_view_net, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([1.0, 1.0], requires_grad=True)
    multi_view_copy_backward(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())

    x3 = Tensor([[1., 2., 3.], [2., 1., 3.]])
    y3 = Tensor([[1.0], [2.0]])
    sens = Tensor([[1., 1., 1.], [1., 1., 1.]])
    multi_view_net2 = MultiViewCopyNet2()
    multi_view_net2.set_inputs()
    grad_fn3 = GradOfAllInputs(multi_view_net2, sens_param=True)
    grads3 = grad_fn3(x3, y3, sens)
    x_torch3 = torch.tensor([[1., 2., 3.], [2., 1., 3.]], requires_grad=True)
    y_torch3 = torch.tensor([[1.0], [2.0]], requires_grad=True)
    multi_view_copy_backward2(x_torch3, y_torch3)
    np.testing.assert_almost_equal(grads3[0].asnumpy(), x_torch3.grad.numpy())
    np.testing.assert_almost_equal(grads3[1].asnumpy(), y_torch3.grad.numpy())

    x4 = Tensor([[1., 2., 3., 4., 4.],
                 [1., 2., 3., 4., 4.],
                 [1., 2., 3., 0., 0.],
                 [1., 2., 5., 4., 4.],
                 [1., 2., 2., 4., 4.]])
    y4 = Tensor([[1., 2., 3., 4., 4.],
                 [1., 2., 3., 4., 4.],
                 [1., 2., 3., 0., 0.],
                 [1., 2., 1., 4., 4.],
                 [1., 1., 1., 4., 4.]])
    z4 = Tensor([[1., 2.], [2., 1.]])
    sens = Tensor([[1., 1.], [1., 1.]])
    multi_view_net3 = MultiViewCopyNet3()
    multi_view_net3.set_inputs()
    grad_fn3 = GradOfAllInputs(multi_view_net3, sens_param=True)
    grads4 = grad_fn3(x4, y4, z4, sens)
    x_torch4 = torch.tensor([[1., 2., 3., 4., 4.], [1., 2., 3., 4., 4.],
                             [1., 2., 3., 0., 0.], [1., 2., 5., 4., 4.],
                             [1., 2., 2., 4., 4.]], requires_grad=True)
    y_torch4 = torch.tensor([[1., 2., 3., 4., 4.], [1., 2., 3., 4., 4.],
                             [1., 2., 3., 0., 0.], [1., 2., 1., 4., 4.],
                             [1., 1., 1., 4., 4.]], requires_grad=True)
    z_torch4 = torch.tensor([[1., 2.], [2., 1.]], requires_grad=True)
    multi_view_copy_backward3(x_torch4, y_torch4, z_torch4)
    np.testing.assert_almost_equal(grads4[0].asnumpy(), x_torch4.grad.numpy())
    np.testing.assert_almost_equal(grads4[1].asnumpy(), y_torch4.grad.numpy())
    np.testing.assert_almost_equal(grads4[2].asnumpy(), z_torch4.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_multi_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    z1 = Tensor([[3.0], [3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0]])
    view_copy_net = MultiCopyViewNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=True)
    grads1 = grad_fn1(x1, y1, z1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    z_torch1 = torch.tensor([[3.0], [3.0]], requires_grad=True)
    multi_copy_view_backward(x_torch1, y_torch1, z_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[2].asnumpy(), z_torch1.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([1., 2., 3.])
    sens = Tensor([1., 1., 1.])
    view_copy_net = ConstantTensorCopyViewNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=True)
    grads1 = grad_fn1(x1, sens)
    x_torch1 = torch.tensor([1., 2., 3.], requires_grad=True)
    constant_tensor_copy_view_net(x_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_as_strided_overlap_grad():
    """
    Feature: Test as strided grad.
    Description: test as strided grad.
    Expectation: with valid exception.
    """
    x1 = Tensor([1., 2., 3.])
    sens = Tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    as_strided_net = AsStridedExpandNet()
    as_strided_net.set_inputs()
    grad_fn1 = GradOfAllInputs(as_strided_net, sens_param=True)
    grads1 = grad_fn1(x1, sens)
    x_torch1 = torch.tensor([1., 2., 3.], requires_grad=True)
    as_strided_expand(x_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())

    x_torch2 = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13, 14, 15], requires_grad=True)
    as_strided_overlap(x_torch2)
    x2 = Tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13, 14, 15])
    sens = Tensor([[1., 1.], [1., 1.], [1., 1.]])
    overlap_net = AsStridedOverlapNet()
    overlap_net.set_inputs()
    grad_fn2 = GradOfAllInputs(overlap_net, sens_param=True)
    grads2 = grad_fn2(x2, sens)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())

    x_torch3 = torch.tensor([[1.0], [4.0], [7.0]], requires_grad=True)
    as_strided_input_overlap(x_torch3)
    x3 = Tensor([[1.0], [4.0], [7.0]])
    sens = Tensor([[1.0]])
    inputoverlap_net = AsStridedInputOverlapNet()
    inputoverlap_net.set_inputs()
    grad_fn3 = GradOfAllInputs(inputoverlap_net, sens_param=True)
    grads3 = grad_fn3(x3, sens)
    np.testing.assert_almost_equal(grads3[0].asnumpy(), x_torch3.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_with_bprop():
    """
    Feature: Test view inplace grad with bprop.
    Description: test view copy grad with bprop.
    Expectation: No exception.
    """
    x1_np = np.array([[1.0, 3.0, 4.0], [2.0, 2.0, 2.0], [2.0, 3.0, 4.0]]).astype(np.float32)
    y1_np = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]).astype(np.float32)
    x1_ms = Tensor(x1_np)
    y1_ms = Tensor(y1_np)
    sens = Tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    net1 = MsViewInplaceHasBprop()
    net1.set_inputs()
    grad_fn1 = GradOfAllInputs(net1, sens_param=True)
    grads1 = grad_fn1(x1_ms, y1_ms, sens)

    x1_torch = torch.tensor(x1_np, requires_grad=True)
    y1_torch = torch.tensor(x1_np, requires_grad=True)
    out_torch = TorchViewInplaceHasBprop.apply(x1_torch, y1_torch)
    out_torch.sum().backward()
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x1_torch.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y1_torch.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_grad_check_exception():
    """
    Feature: Test view inplace valid.
    Description: test view inplace valid
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2., 2., 2.], [3., 3., 3.]])
    y1 = Tensor([[1., 2.], [2., 2.]])
    sens = Tensor([[1., 1.], [1., 1.]])
    no_grad_view_net = NoGradViewCopyNet()
    no_grad_view_net.set_inputs()
    grad_fn1 = GradOfAllInputs(no_grad_view_net, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn1(x1, y1, sens)
        _pynative_executor.sync()
    assert "which created in no_grad mode and inplace modified with grad mode enabled" in str(err.value)

    x2 = Tensor([[1., 2.], [2., 2.], [3., 3.], [1, 1.]])
    y2 = Tensor([[1., 2.], [2., 2.]])
    sens = Tensor([[1., 1.], [1., 1.]])
    multi_output_view_net = MultiOutputViewCopyNet()
    multi_output_view_net.set_inputs()
    grad_fn2 = GradOfAllInputs(multi_output_view_net, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn2(x2, y2, sens)
        _pynative_executor.sync()
    assert "This view is one of output for multi output operator" in str(err.value)

    x3 = Tensor([[2., 3.], [2., 2.], [3., 3.]])
    y3 = Tensor([[1., 2.], [2., 2.], [1, 1]])
    sens = Tensor([[1., 1.], [1., 1.], [1, 1]])
    leaf_view_copy = LeafViewCopyNet()
    leaf_view_copy.set_inputs()
    grad_fn3 = GradOfAllInputs(leaf_view_copy, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn3(x3, y3, sens)
        _pynative_executor.sync()
    assert "A leaf tensor that requires grad is being used in an inplace operator" in str(err.value)

    x4 = Tensor([[2.], [2.], [3.]])
    y4 = Tensor([[1., 2.], [2., 2.], [1, 1]])
    z4 = mint.broadcast_to(x4, (3, 2))
    overlap_view_copy = OverlapViewCopyNet()
    with pytest.raises(RuntimeError) as err:
        overlap_view_copy.construct(z4, y4)
        _pynative_executor.sync()
    assert "This tensor has multi element reference to the same memory address" in str(err.value)

    x5 = Tensor([[1., 2.], [2., 2.], [3., 3.], [1, 1.]])
    y5 = Tensor([[1., 2.]])
    sens = Tensor([[1., 1.]])
    multi_output_view_net = MultiOutputViewCopyNet2()
    grad_fn2 = GradOfAllInputs(multi_output_view_net, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn2(x5, y5, sens)
        _pynative_executor.sync()
    assert "This view is one of output for multi output operator" in str(err.value)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_triggers_requires_grad_propagation():
    """
    Feature: Test view inplace valid.
    Description: Test whether inplace update on a base tensor makes its view require grad.
    Expectation: The calculation result is correct.
    """

    def fn(y):
        x = Tensor([1.0, 2.0, 3.0], dtype=ms.float32)
        x_view = x[:2]
        x[0] = y[0]
        return x_view * x_view

    input_tensor = Tensor([2.0, 2.0])
    grad_op = GradOfFirstInput(fn, sens_param=False)
    grad = grad_op(input_tensor)
    assert np.allclose(grad.asnumpy(), np.array([4.0, 0.0], dtype=np.float32), 0.000001, 0.000001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_on_view_base():
    """
    Feature: Test view inplace valid.
    Description:  Verify view + inplace mechanism when base tensor is also a view tensor.
    Expectation: The calculation result is correct.
    """

    def fn(input_tensor):
        input_view = input_tensor[1]
        with _no_grad():
            input_tensor += 1.0
        return input_view

    data = Tensor([[1.0, 2.0], [1.0, 2.0]])
    input_tensor = ops.stop_gradient(data[1])
    grad_op = ops.GradOperation(get_all=True)
    grad = grad_op(fn)(input_tensor)
    np.allclose(grad[0].asnumpy(), np.array([0.0, 1.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_with_unsafe_view():
    """
    Feature: Test view inplace valid.
    Description: Test reshape ops.
    Expectation: The calculation result is correct.
    """

    def fn(input_tensor):
        x = input_tensor + 1.0
        x_t = x.transpose(1, 0)
        x_r = x_t.reshape(1, 4)
        x_r_v = x_r[0][1]
        x_r_v.mul_(2.0)
        return x_r_v

    input_tensor = Tensor(([1.0, 2.0], [1.0, 2.0]))
    grad_op = GradOfFirstInput(fn, sens_param=False)
    grad = grad_op(input_tensor)
    assert np.allclose(grad.asnumpy(), np.array([[0., 0.], [2., 0.]], dtype=np.float32), 0.000001, 0.000001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_with_multiout_view_and_version_update():
    """
    Feature: Test view inplace valid.
    Description: Test version update of multi output view.
    Expectation: The calculation result is correct.
    """

    def fn(input_tensor):
        x = input_tensor + 1.0
        x.mul_(2.0)
        y = x.split((1, 1), 0)
        z = y[0].mul(2.0)
        return z

    input_tensor = Tensor(([1.0, 2.0], [1.0, 2.0]))
    grad_op = GradOfFirstInput(fn, sens_param=False)
    grad = grad_op(input_tensor)
    assert np.allclose(grad.asnumpy(), np.array([[4., 4.], [0., 0.]], dtype=np.float32), 0.000001, 0.000001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_view_rebase_error():
    """
    Feature: Test view inplace valid.
    Description: Test view rebase error.
    Expectation: Raise RuntimeError.
    """

    def fn(x):
        x_view1, x_view2 = ops.split(x, 1)
        with _no_grad():
            x_view1 += 1.0
        return x_view2

    input_tensor = Tensor([[1., 2.], [2., 2.]])

    grad_op = GradOfFirstInput(fn, sens_param=False)
    with pytest.raises(RuntimeError) as err:
        grad_op(input_tensor)
        _pynative_executor.sync()
    assert "A view of base is being rebase" in str(err.value)
    assert "This view is one of output for multi output operator" in str(err.value)

    def fn1(x):
        y = x * 2.0
        with _no_grad():
            y_view = y[0]
        y += 1.0
        return y_view

    grad_op = GradOfFirstInput(fn1, sens_param=False)
    with pytest.raises(RuntimeError) as err:
        grad_op(input_tensor)
        _pynative_executor.sync()
    assert ("A view of base is being rebase, "
            "which created in no_grad mode and inplace modified with grad mode enabled.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_base_tensor_is_no_requires_grad_and_view_tensor_is_leaf():
    """
    Feature: Test view inplace valid.
    Description: When the base tensor of a view does not require gradient and
                 the view tensor is a leaf node requiring gradient, version and
                 creation_type check should not be triggered.
    Expectation: Not raise error and the calculation result is correct.
    """
    x = ms.Tensor([2.0, 1.0])
    x_view = x.view_as(x)
    x_view += 1.0

    # view tensor is leaf tensor
    grad = ms.grad(lambda input_tensor: input_tensor * input_tensor, grad_position=0)(x_view)
    assert np.allclose(grad.asnumpy(), np.array([6.0, 4.0], dtype=np.float32), 0.000001, 0.000001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_base_tensor_is_no_requires_grad_and_view_tensor_is_non_leaf():
    """
    Feature: Test view inplace valid.
    Description: When the base tensor of a view does not require gradient and
                 the view tensor is a non-leaf node requiring gradient, version and
                 creation_type check should be triggered.
    Expectation: Raise runtime error.
    """

    def fn(x):
        x_view1, _ = ops.split(x, 1)
        ms.ops.stop_gradient_(x)
        x += 1.0
        return x_view1

    x = ms.Tensor([[3., 2.], [2., 1.]])
    with pytest.raises(RuntimeError) as err:
        ms.grad(fn, grad_position=0)(x)
        ms.runtime.synchronize()
    assert "A view of base is being rebase" in str(err.value)
    assert "This view is one of output for multi output operator" in str(err.value)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_view_grad_node_record_ouptuts_tensor_meta_data():
    """
    Feature: Test view grad valid.
    Description: Verify if the view grad node records the tensor meta data of outputs.
    Expectation: The calculation result is correct.
    """
    def fn(input_tensor):
        y = input_tensor.split((2, 3), -1)
        z = input_tensor.transpose(-1, -2)
        return y[0], y[1], z

    input_tensor = Tensor(np.random.randn(2, 5).astype(np.float32))
    grad_op = GradOfFirstInput(fn)
    grads = (Tensor(np.ones((2, 2)).astype(np.float32)),
             Tensor(np.ones((2, 3)).astype(np.float32)),
             Tensor(np.ones((5, 2)).astype(np.float32)))
    grad = grad_op(input_tensor, grads)
    expect_grad = np.broadcast_to(np.array(2.).astype(np.float32), (2, 5))
    assert np.allclose(grad.asnumpy(), expect_grad)
