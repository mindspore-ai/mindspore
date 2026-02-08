# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test_custom_function """

import numpy as np
import pytest
import mindspore
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore import _Function
from tests.mark_utils import arg_mark

oneslike = P.OnesLike()


class MultiInputFunctionNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z

    @staticmethod
    def backward(ctx, z):
        return z * 3, z * 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input():
    """
    Feature: Custom autograd function.
    Description: Custom forward function of multi input single output.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MutiInputFunctionErrorNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z

    @staticmethod
    def backward(ctx, z):
        return z * 3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_grad_num_wrong():
    """
    Feature: Custom autograd function.
    Description: The size of backward return is not eq to the size of forward inputs.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MutiInputFunctionErrorNet.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "Function backward return a wrong number of gradients" in str(err.value)


class MultiInputMultiOutputFunctionNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z, t

    @staticmethod
    def backward(ctx, z, t):
        return z * 3, z * 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_multi_output():
    """
    Feature: Custom autograd function.
    Description: Custom forward function of multi input multi output.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MultiInputMultiOutputStarArgsFunctionNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z, t

    @staticmethod
    def backward(ctx, *args):
        return args[0] * 3, args[1] * 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_multi_output_star_args():
    """
    Feature: Custom autograd function.
    Description: Custom forward function of multi input multi output, backward input is *args.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputStarArgsFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MultiInputMultiOutputNotTensorFunctionNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z, 4, t

    @staticmethod
    def backward(ctx, *args):
        return args[0] * 3, args[2] * 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_multi_output_not_tensor():
    """
    Feature: Custom autograd function.
    Description: Some output of backward function is not tensor.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputNotTensorFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MultiInputMultiOutputFunctionErrorNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z, t

    @staticmethod
    def backward(ctx, z):
        return z * 3, z * 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_backward_input_num_wrong():
    """
    Feature: Custom autograd function.
    Description: The size of backward inputs is not equal to the size of forward outputs.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputFunctionErrorNet.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(TypeError):
        grad_net(net)(x, y)


class MultiInputMultiOutputFunctionError1Net(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z, t

    @staticmethod
    def backward(ctx, z, t):
        return z * 3, z * 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_no_tensor_grad_should_be_none():
    """
    Feature: Custom autograd function.
    Description: When the input of forward is not tensor, the corresponding output of backward should be none.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = 3
    net = MultiInputMultiOutputFunctionError1Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "Input is not tensor, but gradient is not none" in str(err.value)


class MultiInputMultiOutputFunctionError2Net(_Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x * x
        z = t + y
        return z, t

    @staticmethod
    def backward(ctx, z, t):
        return z * 3, 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_grad_should_be_none_or_tensor():
    """
    Feature: Custom autograd function.
    Description: The outputs of backward should be nont or tensor.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputFunctionError2Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "Gradient should be none or tensor" in str(err.value)


class CustomFunctionContextNet(_Function):
    @staticmethod
    def forward(ctx, x):
        ctx.age = 7
        x2 = x * x
        y = x2 + 1
        ctx.save_for_backward(x, x2, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, x2, y = ctx.saved_tensors
        age = ctx.age
        assert np.allclose(x.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
        assert np.allclose(x2.asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
        assert np.allclose(y.asnumpy(), np.array([5], dtype=np.float32), 0.00001, 0.00001)
        assert age == 7, "age should be 7."
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_context():
    """
    Feature: Custom autograd function.
    Description: The context can bring information from forward function to backward function.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    net = CustomFunctionContextNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNeedGradNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        return z

    @staticmethod
    def backward(ctx, grad_output):
        need_grad = ctx.needs_input_grad
        assert len(need_grad) == 2, "number of need grad should be same as input size."
        assert need_grad[0], "first input need grad"
        assert not need_grad[1], "second input do not need grad"
        return grad_output, None


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_need_grad():
    """
    Feature: Custom autograd function.
    Description: User can get the information of whether the inputs of forward function need grad.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = 3
    net = CustomFunctionNeedGradNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNeedGradForwardNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        need_grad = ctx.needs_input_grad
        assert len(need_grad) == 2, "number of need grad should be same as input size."
        assert need_grad[0], "first input need grad"
        assert not need_grad[1], "second input do not need grad"
        x2 = x * x
        z = x2 + 1
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_need_grad_forward():
    """
    Feature: Custom autograd function.
    Description: User can get the information of whether the inputs of forward function need grad.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = 3
    net = CustomFunctionNeedGradForwardNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionDirtyTensorError1Net(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x.add_(1)
        ctx.mark_dirty(x)
        z = x + y
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * 2


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.0)
def test_custom_function_dirty_tensor_must_all_be_output():
    """
    Feature: Custom autograd function.
    Description: Throw runtime error when exit dirty tensor is not output of forward function.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionDirtyTensorError1Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "The dirty tensors must all be outputs of the forward function." in str(err.value)


class CustomFunctionDirtyTensorError2Net(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x.add_(1)
        ctx.mark_dirty(x)
        z = x + y
        return z, x

    @staticmethod
    def backward(ctx, grad_output, a):
        return grad_output, grad_output * 2


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_custom_function_dirty_tensor_is_need_grad_leaf():
    """
    Feature: Custom autograd function.
    Description: Throw runtime error when leaf tensor need grad is modified.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionDirtyTensorError2Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "A leaf tensor that need grad is being used in an inplace operator" in str(err.value)


class CustomFunctionDirtyTensorNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x.add_(1)
        ctx.mark_dirty(x)
        z = x + y
        return z, x

    @staticmethod
    def backward(ctx, grad_output, a):
        return grad_output, grad_output * 2


class CustomFunctionLeafNet(nn.Cell):
    def construct(self, y):
        x = Tensor([2, 2], mindspore.float32)
        return CustomFunctionDirtyTensorNet.apply(x, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_custom_function_dirty_tensor_not_leaf_no_grad():
    """
    Feature: Custom autograd function.
    Description: Input tensor of forward function is not leaf, no need to do grad, can be modified.
    Expectation: success.
    """
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionLeafNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(y)
    assert np.allclose(grads[0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionLeafNet1(nn.Cell):
    def construct(self, x, y):
        x = x + 1
        return CustomFunctionDirtyTensorNet.apply(x, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_custom_function_dirty_tensor_not_leaf_need_grad():
    """
    Feature: Custom autograd function.
    Description: Input tensor of forward function is not leaf, need to do grad, can be modified.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionLeafNet1()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class InplaceMulOp(_Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x.clone(), y)
        ctx.mark_dirty(x)
        return x.mul_(y)

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        grad_x = grad_out * y if ctx.needs_input_grad[0] else None
        grad_y = grad_out * x if ctx.needs_input_grad[1] else None
        return grad_x, grad_y


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.0)
def test_custom_function_dirty_tensor_is_view():
    """
    Feature: Custom autograd function.
    Description: Input tensor of forward function is not leaf and is a view tensor, need to do grad, can be modified.
    Expectation: success.
    """
    x = Tensor([2.0, 1.0], mindspore.float32)
    y = Tensor([3.0, 2.0], mindspore.float32)

    def fn(x, y):
        z = x + y
        return InplaceMulOp.apply(z[0], y[0])

    grad_op = ops.GradOperation(get_all=True)(fn)
    grad_x, grad_y = grad_op(x, y)
    assert np.allclose(grad_x.asnumpy(), np.array([3.0, 0.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), np.array([8.0, 0.0], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNotLeafNet(nn.Cell):
    def construct(self, y):
        x = Tensor([2., 2, 2], mindspore.float32)
        x1 = x[1:3]
        return CustomFunctionDirtyTensorNet.apply(x1, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_custom_function_dirty_view_multi_output_error():
    """
    Feature: Custom autograd function.
    Description: The scene dirty tensor is view and forward function is multi-output is prohibited.
    Expectation: success.
    """
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNotLeafNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(y)
    assert "A view is one of output for multi output operator" in str(err.value)


class CustomFunctionOutIsInNet(_Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 2


class CustomFunctionWrapOutIsInNet(nn.Cell):
    def construct(self, y):
        y = ops.split(y, 1)[1]
        return CustomFunctionOutIsInNet.apply(y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_output_is_input():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is input.
    Expectation: success.
    """
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionWrapOutIsInNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(y)
    assert np.allclose(grads[0].asnumpy(), np.array([0, 2.], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
def test_custom_function_output_is_input_inplace_error():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is input, and perform inplace ops on output.
    Expectation: Raise RuntimeError.
    """

    def fn(x):
        y = CustomFunctionOutIsInNet.apply(x)
        y.add_(1.0)
        return y

    x = Tensor([3, 3], mindspore.float32)
    grad_op = ops.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_op(fn)(x)
    assert ("This view tensor is output of custom cell, which has custom bprop, "
            "it may not support view+inplace") in str(err.value)


class CustomFunctionNonDiffNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.mark_non_differentiable(z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_non_diff():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is marked non-diff.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNonDiffNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNonDiffNet1(_Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.mark_non_differentiable(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_non_diff1():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is marked non-diff when output is input.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNonDiffNet1.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionViewNotInAndDirtyNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.save_for_backward(x, y)
        w = z[0:1]
        return w

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return ops.OnesLike()(x), ops.OnesLike()(y) * 2


class CustomFunctionViewNotInAndDirtyWrapNet(nn.Cell):
    def construct(self, x, y):
        out = CustomFunctionViewNotInAndDirtyNet.apply(x, y)
        out.add_(1)
        return out


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.0)
def test_custom_function_view_not_in_and_dirty():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is a view, and is modified subsequently.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionViewNotInAndDirtyWrapNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "This view tensor is output of custom cell" in str(err.value)


class CustomFunctionMultiDiffOutputNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.save_for_backward(x, y)
        v1 = z[0:1]
        return v1, x2

    @staticmethod
    def backward(ctx, grad_output, grad_output1):
        x, y = ctx.saved_tensors
        return ops.OnesLike()(x), ops.OnesLike()(y) * 2


class CustomFunctionMultiDiffOutputWrapNet(nn.Cell):
    def construct(self, x, y):
        out = CustomFunctionMultiDiffOutputNet.apply(x, y)
        out[0].add_(1)
        return out


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_custom_function_multi_diff_output():
    """
    Feature: Custom autograd function.
    Description: The output of multi-output forward function is a view, and is modified subsequently.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionMultiDiffOutputWrapNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "A view of base is being inplace modified" in str(err.value)


class CustomFunctionMaterializeGradsNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        return z, x2

    @staticmethod
    def backward(ctx, grad_output, grad_output1):
        grad_output2 = grad_output1 + 2
        return grad_output, grad_output2


class CustomFunctionMaterializeGradsWrap(nn.Cell):
    def construct(self, x, y):
        z, _ = CustomFunctionMaterializeGradsNet.apply(x, y)
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_materialize_grad():
    """
    Feature: Custom autograd function.
    Description: None output grad tensor should be materialized default.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionMaterializeGradsWrap()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNotMaterializeGradsNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.set_materialize_grads(False)
        return z, x2

    @staticmethod
    def backward(ctx, grad_output, grad_output1):
        with pytest.raises(TypeError):
            grad_output = grad_output1 + 2
        return grad_output, grad_output


class CustomFunctionNotMaterializeGradsWrap(nn.Cell):
    def construct(self, x, y):
        z, _ = CustomFunctionNotMaterializeGradsNet.apply(x, y)
        return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_not_materialize_grad():
    """
    Feature: Custom autograd function.
    Description: None output grad tensors are not materialized.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNotMaterializeGradsWrap()
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x, y)


class CustomFunctionWithAttr(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        x.tensor = x2
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x)
        return z, x2

    @staticmethod
    def backward(ctx, grad_output, grad_output1):
        x, = ctx.saved_tensors
        x2 = x.tensor
        return grad_output, x2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_with_attr():
    """
    Feature: Custom autograd function.
    Description: None output grad tensors with attr.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionWithAttr()
    grad_net = C.GradOperation(get_all=True)
    grad_net(net.apply)(x, y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_reduce_same_shape():
    """
    Feature: Custom autograd function same shape.
    Description: Test auto reduce.
    Expectation: success.
    """

    class CustomFunctionAutoReduceNet(_Function):
        @staticmethod
        def forward(ctx, x, y):
            x2 = x + y
            return x2

        @staticmethod
        def backward(ctx, *args):
            return Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]]), \
                Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]])

    x = Tensor([3, 3, 3], mindspore.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], mindspore.float32)
    net = CustomFunctionAutoReduceNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net.apply)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_reduce_not_same_shape():
    """
    Feature: Custom autograd function no same shape.
    Description: Test auto reduce.
    Expectation: success.
    """

    class CustomFunctionAutoReduceNet2(_Function):
        @staticmethod
        def forward(ctx, x, y):
            x2 = x + y
            return x2

        @staticmethod
        def backward(ctx, *args):
            return Tensor(np.ones((1, 1, 24, 8828, 128)).astype('float32')), \
                Tensor(np.ones((1, 1, 24, 8828, 128)).astype('float32'))

    x = Tensor(np.random.rand(1, 24, 8828, 128).astype('float32'))
    y = Tensor(np.random.rand(1, 24, 8828, 128).astype('float32'))
    net = CustomFunctionAutoReduceNet2()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net.apply)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.ones((1, 24, 8828, 128), dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.ones((1, 24, 8828, 128), dtype=np.float32),
                       0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_reduce_zero_shape():
    """
    Feature: Custom autograd function zero shape.
    Description: Test auto reduce.
    Expectation: success.
    """

    class CustomFunctionAutoReduceNet2(_Function):
        @staticmethod
        def forward(ctx, x, y):
            x2 = x + y
            return x2

        @staticmethod
        def backward(ctx, *args):
            return Tensor(np.ones((1, 128)).astype('float32')), Tensor(np.ones((1, 128)).astype('float32'))

    x = Tensor(1.)
    y = Tensor(2.)
    net = CustomFunctionAutoReduceNet2()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net.apply)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array(128.), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array(128.), 0.00001, 0.00001)


class CustomFunctionAutoCastNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x + y
        return x2

    @staticmethod
    def backward(ctx, *args):
        return Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=mindspore.int64), \
            Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=mindspore.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_cast():
    """
    Feature: Custom autograd function.
    Description: Test auto cast.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], mindspore.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], mindspore.float32)
    net = CustomFunctionAutoCastNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net.apply)(x, y)
    assert grads[0].dtype == mindspore.float32
    assert grads[1].dtype == mindspore.float32
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionBroadcastExecptionNet(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x + y
        return x2

    @staticmethod
    def backward(ctx, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=mindspore.int64), \
            Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=mindspore.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_reduce_exception():
    """
    Feature: Custom autograd function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], mindspore.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], mindspore.float32)
    net = CustomFunctionBroadcastExecptionNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net.apply)(x, y)
    assert "grad tensor should be broadcast to" in str(err.value)


class CustomFunctionSelfRequiresGrad(_Function):
    @staticmethod
    def forward(ctx, x, y):
        x._requires_grad = True
        return x

    @staticmethod
    def backward(ctx, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=mindspore.int64), \
            Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=mindspore.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_self_requires_grad():
    """
    Feature: Custom autograd function.
    Description: Test self requires grad.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], mindspore.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], mindspore.float32)
    z = CustomFunctionSelfRequiresGrad.apply(x, y)
    assert x._requires_grad is True
    assert z._requires_grad is False


class CustomFunctionOutRequiresGrad(_Function):
    @staticmethod
    def forward(ctx, x, y):
        z = Tensor([3, 3, 3], mindspore.float32)
        z._requires_grad = True
        return z

    @staticmethod
    def backward(ctx, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=mindspore.int64), \
            Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=mindspore.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_out_requires_grad_true():
    """
    Feature: Custom autograd function.
    Description: Test out requires grad.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], mindspore.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], mindspore.float32)
    z = CustomFunctionOutRequiresGrad.apply(x, y)
    assert x._requires_grad is False
    assert z._requires_grad is False


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_ctx_get_unset_attr():
    """
    Feature: Custom autograd function.
    Description: Verify that all unset attributes of the autograd context (ctx) can be safely accessed.
    Expectation: Not raise error.
    """

    class CustomOp(_Function):
        @staticmethod
        def forward(ctx, x):
            for key in dir(ctx):
                try:
                    getattr(ctx, key)
                except AttributeError as e:
                    print(f'getattr error: {e}')
            return x

    x = mindspore.Tensor(1.0)
    CustomOp.apply(x)


class CustomFunctionNotRaiseError(_Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward((x, y))
        return x + y

    @staticmethod
    def backward(ctx, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=mindspore.int64), \
            Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=mindspore.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_not_raise_error():
    """
    Feature: Custom autograd function.
    Description: Test save tuple not raise error
    Expectation: success.
    """
    x = Tensor([3, 3, 3], mindspore.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], mindspore.float32)
    _ = CustomFunctionNotRaiseError.apply(x, y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_mark_non_differentiable_error():
    """
    Feature: Custom autograd function.
    Description: Test mark_non_differentiable for non-tensor.
    Expectation: Raise Runtime error.
    """

    class MarkNonDiffErrorOp(_Function):
        @staticmethod
        def forward(ctx, x):
            ctx.mark_non_differentiable(x, 1.0)
            return x

    with pytest.raises(RuntimeError, match="element of non_differentiable should be a tensor"):
        MarkNonDiffErrorOp.apply(Tensor([1.0]))


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_mark_dirty_error():
    """
    Feature: Custom autograd function.
    Description: Test mark_dirty for non-tensor.
    Expectation: Raise Runtime error.
    """

    class MarkDirtyErrorOp(_Function):
        @staticmethod
        def forward(ctx, x):
            ctx.mark_dirty(x, 1.0)
            return x

    with pytest.raises(RuntimeError, match="element of dirty_tensors should be a tensor"):
        MarkDirtyErrorOp.apply(Tensor([1.0]))


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_grad_node():
    """
    Feature: Custom autograd function.
    Description: Test grad node saved tensors
    Expectation: success.
    """
    class CustomFunctionContextCell(nn.Cell):
        def construct(self, x):
            return CustomFunctionContextNet.apply(x)

    x = Tensor([3, 3, 3], mindspore.float32)
    forward_net = CustomFunctionContextCell()
    forward_net.set_grad(True)
    output = forward_net(x)
    assert len(output._grad_node.next_functions) == 1
    assert np.allclose(output._grad_node.saved_tensors[0].asnumpy(), np.array([3., 3., 3.]), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_add_saved_tensor_not_tensor_or_none():
    """
    Feature: Custom autograd function.
    Description: Test saved tensor
    Expectation: Raise Runtime error.
    """

    class RaiseErrorNet(_Function):
        @staticmethod
        def forward(ctx, x):
            ctx.age = 7
            x2 = x * x
            y = x2 + 1
            ctx.save_for_backward(x, (x2, y))
            return y

        @staticmethod
        def backward(ctx, grad_output):
            _ = ctx.saved_tensors
            return grad_output

    x = Tensor([3, 3, 3], mindspore.float32)
    with pytest.raises(RuntimeError, match="only support None and tensor"):
        _ = mindspore.grad(RaiseErrorNet.apply)(x)
