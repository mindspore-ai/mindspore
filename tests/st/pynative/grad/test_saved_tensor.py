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
""" test_saved_tensor """
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, _Function
from mindspore.graph.api import _pynative_executor, _disable_saved_tensors_hooks
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, GradOfFirstInput, GradOfAllParams


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_version_check():
    """
    Features: Saved Tensor Version check.
    Description: Test inplace modification on saved tensor triggers version check error.
    Expectation: Raise RuntimeError with version mismatch message.
    """
    x = Tensor([2.0, 3.0], dtype=ms.float32)
    y = Tensor(2.0, dtype=ms.float32)

    def mul_fn(x, y):
        z = y * 2.0
        out = x * z
        z += 1.0
        return out

    grad_op = ops.GradOperation(get_all=True)(mul_fn)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*input"):
        grad_op(x, y)

    def inplace_mul_fn(x):
        z = x + 1.0
        z *= z
        return z

    grad_op = ops.GradOperation(get_all=True)(inplace_mul_fn)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*input"):
        grad_op(y)

    def relu_fn(x):
        out = ops.relu(x)
        out += x
        return out

    grad_op = ops.GradOperation(get_all=True)(relu_fn)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*output"):
        grad_op(x)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_custom_function_saved_tensor_version_check():
    """
    Features: Saved Tensor Version check.
    Description: Test inplace modification on the tensor which saved in custom function triggers version check error.
    Expectation: Raise RuntimeError with version mismatch message.
    """

    class CustomOp(_Function):
        @staticmethod
        def forward(ctx, x, y):
            out1 = x + y
            out2 = x * y
            ctx.save_for_backward(out1, x)
            return out1, out2

        @staticmethod
        def backward(ctx, grad_out1, grad_out2):
            out1, x = ctx.saved_tensors
            return x * grad_out1, out1 * grad_out2

    x = Tensor([1.0, 3.0], dtype=ms.float32)
    y = Tensor([2.0, 3.0], dtype=ms.float32)

    def custom_fn_inplace_saved_output_tensor(x, y):
        z1, z2 = CustomOp.apply(x, y)
        z1 *= 2.0
        return z1 + z2

    grad_op = ops.GradOperation(get_all=True)(custom_fn_inplace_saved_output_tensor)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*custom"):
        grad_op(x, y)

    def custom_fn_inplace_saved_input_tensor(x, y):
        x = x + 1.0
        z1, z2 = CustomOp.apply(x, y)
        x += 1.0
        return z1 + z2

    grad_op = ops.GradOperation(get_all=True)(custom_fn_inplace_saved_input_tensor)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*custom"):
        grad_op(x, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_custom_bprop_cell_saved_tensor_version_check():
    """
    Features: Saved Tensor Version check.
    Description: Test inplace modification on the tensor which saved in custom bprop cell triggers version check error.
    Expectation: Raise RuntimeError with version mismatch message.
    """

    class CustomSavedBpropCell(nn.Cell):
        def construct(self, x, y):
            z = x * y
            return ops.relu(z)

        def bprop(self, x, y, _, grad):
            return grad * y, grad * x

    net = CustomSavedBpropCell()
    x = Tensor([1.0, 3.0], dtype=ms.float32)
    y = Tensor([2.0, 3.0], dtype=ms.float32)

    def forward_fn_inplace_saved_input(x, y):
        y = y + 1.0
        out = net(x, y)
        y += 1.0
        return out

    grad_op = ops.GradOperation(get_all=True)(forward_fn_inplace_saved_input)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*input"):
        grad_op(x, y)

    def forward_fn_inplace_saved_output(x, y):
        out = net(x, y)
        out += 1.0
        return out

    grad_op = ops.GradOperation(get_all=True)(forward_fn_inplace_saved_output)
    with pytest.raises(RuntimeError, match="modified by an inplace operation.*output"):
        grad_op(x, y)

    net.used_bprop_inputs = [0, 1]
    grad_op = ops.GradOperation(get_all=True)(forward_fn_inplace_saved_output)
    grad_x, grad_y = grad_op(x, y)
    assert np.allclose(grad_x.asnumpy(), y.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), x.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_inplace_detach():
    """
    Features: Saved Tensor.
    Description: Test inplace detach on saved tensors.
    Expectation: Leaf inplace detach succeeds, while intermediate inplace detach raises RuntimeError.
    """

    # leaf node
    def leaf_fn(x, y):
        z = x * y
        ops.stop_gradient_(x)
        return z

    def inter_fn(x, y):
        x_relu = ms.mint.functional.relu(x)
        out = x_relu * y
        ops.stop_gradient_(x_relu)
        return out

    x = ops.rand(5, dtype=ms.float32)
    y = ops.rand(5, dtype=ms.float32)

    # leaf node inplace detach not raise error
    _, grad_y = ms.grad(leaf_fn, grad_position=(0, 1))(x, y)
    np.allclose(grad_y.asnumpy(), x.asnumpy(), 0.00001, 0.00001)

    with pytest.raises(RuntimeError, match="Trying to use a saved tensor that has been detached in-place"):
        ms.grad(inter_fn, grad_position=(0, 1))(x, y)


class DelRecordHook:
    record_list = []

    def __call__(self, unused_grad):
        return

    def __del__(self):
        self.record_list.append(0)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_op_output_no_cycle():
    """
    Features: Saved Tensor.
    Description: Test that saving operator outputs does not cause circular references
                 even when the backward node is not executed.
    Expectation: All hooks are released normally and no circular references occur.
    """

    class CustomSavedOutputOp(_Function):
        @staticmethod
        def forward(ctx, x):
            out1 = x + x
            out2 = x * x
            ctx.save_for_backward(out1, out2)
            return out1, out2

    class CustomSavedOutputBpropCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.used_bprop_inputs = [2]  # only save output

        def construct(self, x, y):
            return x + y, x * y

        def bprop(self, *args):
            out, dout = args[-2:]
            return dout[0] * out[0], dout[1] + out[1]

    custom_bprop_net = CustomSavedOutputBpropCell()
    DelRecordHook.record_list.clear()

    def forward_fn(x):
        x1 = ops.relu(x)  # save single output tensor
        x1.register_hook(DelRecordHook())

        x2 = x * x  # not save single output tensor
        x2.register_hook(DelRecordHook())

        x3, *_ = ops.split(x, 5)  # not save multi output tensor
        x3.register_hook(DelRecordHook())

        x4, _ = CustomSavedOutputOp.apply(x)  # custom function save output
        x4.register_hook(DelRecordHook())

        x5, _ = custom_bprop_net(x, x)
        x5.register_hook(DelRecordHook())
        return x

    input_x = ops.rand(10, 10, dtype=ms.float32)
    ms.value_and_grad(forward_fn, grad_position=0)(input_x)

    assert DelRecordHook.record_list == [0] * 5


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_view_inplace_no_cycle():
    """
    Features: Saved Tensor.
    Description: Test that saving operator outputs with view in-place operations does not
                 introduce circular references even when the backward node is not executed.
    Expectation: All hooks are released normally and no circular references occur.
    """

    class CustomReluOps(_Function):
        @staticmethod
        def forward(ctx, x):
            ms.mint.functional.relu_(x)
            ctx.mark_dirty(x)
            ctx.save_for_backward(x)
            return x

    DelRecordHook.record_list.clear()

    def forward_view_inplace_fn(x, is_custom_function):
        x_m = x * 1.0
        x_view = x_m.view_as(x_m)
        x_view.register_hook(DelRecordHook())
        if is_custom_function:
            x_vi = CustomReluOps.apply(x_view)
        else:
            x_vi = ms.mint.functional.relu_(x_view)
        x_vi.register_hook(DelRecordHook())
        return x

    input_x = ops.rand(10, 10, dtype=ms.float32)
    ms.value_and_grad(forward_view_inplace_fn, grad_position=0)(input_x, False)
    _pynative_executor.sync()
    assert DelRecordHook.record_list == [0, 0]

    DelRecordHook.record_list.clear()

    input_x = ops.rand(10, 10, dtype=ms.float32)
    ms.value_and_grad(forward_view_inplace_fn, grad_position=0)(input_x, True)
    _pynative_executor.sync()
    assert DelRecordHook.record_list == [0, 0]


class WrapRecordHook:
    pack_record_list = []
    unpack_record_list = []

    def __init__(self, hook, is_pack):
        self.hook = hook
        self.is_pack = is_pack

    def __call__(self, x):
        if self.is_pack:
            self.pack_record_list.append(0)
        else:
            self.unpack_record_list.append(0)
        return self.hook(x)

    @classmethod
    def clear(cls):
        cls.pack_record_list.clear()
        cls.unpack_record_list.clear()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_op_input():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are correctly triggered when operator needs saving inputs.
    Expectation: Success.
    """

    def pack_hook(x):
        return x * 2.0

    def unpack_hook(x):
        return x + 2.0

    class SavedInputNet(nn.Cell):
        def construct(self, x, y):
            return x * y

    WrapRecordHook.clear()

    input_x = ops.rand(2, 3)
    input_y = ops.rand(2, 3)

    net = SavedInputNet()
    net.register_saved_tensors_hooks(WrapRecordHook(pack_hook, True), WrapRecordHook(unpack_hook, False))
    net.set_grad()
    _ = net(input_x, input_y)
    assert WrapRecordHook.pack_record_list == [0, 0]
    assert not WrapRecordHook.unpack_record_list

    grad_x, grad_y = GradOfAllInputs(net, sens_param=False)(input_x, input_y)
    assert WrapRecordHook.unpack_record_list == [0, 0]
    expected_grad_x, expected_grad_y = input_y * 2 + 2.0, input_x * 2 + 2.0
    assert np.allclose(grad_x.asnumpy(), expected_grad_x.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), expected_grad_y.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_op_output():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are correctly triggered when operator needs saving outputs.
    Expectation: Success.
    """

    def pack_hook(x):
        return x + 3.0

    def unpack_hook(x):
        return x

    class SavedInputNet(nn.Cell):
        def construct(self, x):
            return ops.sigmoid(x)

    WrapRecordHook.clear()

    input_x = ops.rand(2, 3)
    net = SavedInputNet()
    net.register_saved_tensors_hooks(WrapRecordHook(pack_hook, True), WrapRecordHook(unpack_hook, False))
    net.set_grad()
    out = net(input_x)
    assert WrapRecordHook.pack_record_list == [0]
    assert not WrapRecordHook.unpack_record_list

    grad_x = GradOfFirstInput(net, sens_param=False)(input_x)
    assert WrapRecordHook.unpack_record_list == [0]
    expected_grad_x = (out + 3.0) * (1 - (out + 3.0))
    assert np.allclose(grad_x.asnumpy(), expected_grad_x.asnumpy(), 0.00001, 0.00001)


def pack_hook_stop_gradient(x):
    return ops.stop_gradient(x)

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_op_input_and_output():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are correctly triggered when operator needs saving inputs and outputs.
    Expectation: Success.
    """

    class SavedInputAndOutputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(4)

        def construct(self, x):
            return self.bn(x)

    WrapRecordHook.clear()

    input_x = ops.rand(2, 4, 32, 32, dtype=ms.float32)
    net = SavedInputAndOutputNet()
    net.register_saved_tensors_hooks(WrapRecordHook(pack_hook_stop_gradient, True),
                                    WrapRecordHook(lambda x: x, False))
    net.set_train()
    net.set_grad()
    net(input_x)
    assert WrapRecordHook.pack_record_list == [0] * 7
    assert not WrapRecordHook.unpack_record_list

    GradOfFirstInput(net, sens_param=False)(input_x)
    assert WrapRecordHook.unpack_record_list == [0] * 7


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_custom_function():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are correctly triggered with custom function.
    Expectation: Raise RuntimeError.
    """

    class CustomSavedFunctionOp(_Function):
        @staticmethod
        def forward(ctx, x, y):
            inter_exp = ops.exp(x)
            out = x + y
            ctx.save_for_backward(inter_exp, out)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            s1, s2 = ctx.saved_tensors
            return grad_out * s1, grad_out * s2

    class CustomSavedFunctionNet(nn.Cell):
        def construct(self, x, y):
            return CustomSavedFunctionOp.apply(x, y)

    WrapRecordHook.clear()

    net = CustomSavedFunctionNet()
    net.register_saved_tensors_hooks(
        WrapRecordHook(lambda x: x + 1.0, True), WrapRecordHook(lambda x: x, False))
    net.set_grad()

    input_x = ops.rand(5, 5, dtype=ms.float32)
    input_y = ops.rand(5, 5, dtype=ms.float32)
    out = net(input_x, input_y)
    assert WrapRecordHook.pack_record_list == [0, 0]
    assert not WrapRecordHook.unpack_record_list

    grad_x, grad_y = GradOfAllInputs(net, sens_param=False)(input_x, input_y)
    assert WrapRecordHook.unpack_record_list == [0, 0]
    expected_grad_x = ops.exp(input_x) + 1.0
    expected_grad_y = out + 1.0
    assert np.allclose(grad_x.asnumpy(), expected_grad_x.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), expected_grad_y.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_custom_bprop_cell():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are correctly triggered with custom bprop cell.
    Expectation: Success.
    """

    class CustomSavedBpropCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.used_bprop_inputs = [0, 2]

        def construct(self, x, y):
            out1 = ops.sigmoid(x) + y
            out2 = ops.relu(y) * x
            return out1, out2

        def bprop(self, x, _, out, dout):
            return dout[0] * out[0] * x, dout[1] * out[1] * x

    WrapRecordHook.clear()

    net = CustomSavedBpropCell()
    net.register_saved_tensors_hooks(
        WrapRecordHook(pack_hook_stop_gradient, True), WrapRecordHook(lambda x: x - 1.0, False))
    net.set_grad()

    input_x = ops.rand(2, 2, dtype=ms.float32)
    input_y = ops.rand(2, 2, dtype=ms.float32)
    out1, out2 = net(input_x, input_y)

    assert WrapRecordHook.pack_record_list == [0] * 3
    assert not WrapRecordHook.unpack_record_list

    grad_x, grad_y = GradOfAllInputs(net, sens_param=False)(input_x, input_y)
    assert WrapRecordHook.unpack_record_list == [0] * 3
    expected_grad_x = (out1 - 1.0) * (input_x - 1.0)
    expected_grad_y = (out2 - 1.0) * (input_x - 1.0)
    assert np.allclose(grad_x.asnumpy(), expected_grad_x.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), expected_grad_y.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_custom_cpp_function():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are correctly triggered with custom cpp function.
    Expectation: Raise RuntimeError.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = ms.Parameter(3.0, requires_grad=True)
            self.my_ops = CustomOpBuilder("my_function_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

        def construct(self, x, y):
            z = self.my_ops.mul(x, self.p)
            return z + y

    WrapRecordHook.clear()
    net = MyNet()
    net.register_saved_tensors_hooks(
        WrapRecordHook(pack_hook_stop_gradient, True), WrapRecordHook(lambda x: x * 2.0, False))
    net.set_grad()

    x = ops.rand(2, 2, dtype=ms.float32)
    y = ops.rand(2, 2, dtype=ms.float32)
    net(x, y)
    assert WrapRecordHook.pack_record_list == [0, 0]
    assert not WrapRecordHook.unpack_record_list

    grad = GradOfAllParams(net, sens_param=False)(x, y)
    assert WrapRecordHook.unpack_record_list == [0, 0]
    assert np.allclose(grad[0].asnumpy(), (x * 2.0).asnumpy())


def err_hook(x):
    raise RuntimeError("this hook should not be called")


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_no_grad():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks are not triggered under no_grad context.
    Expectation: Not raise error.
    """
    x = ms.Parameter(ops.rand(3, 3, dtype=ms.float32), requires_grad=True)
    y = ms.Parameter(ops.rand(3, 3, dtype=ms.float32), requires_grad=True)
    with ms.saved_tensors_hooks(err_hook, err_hook):
        x_r = ms.mint.functional.relu(x)
        _ = x_r * y


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_autograd_saved_tensor_hook_pack_hook_with_inplace_modification_error():
    """
    Features: Saved Tensor Hook.
    Description: Verify that in-place modification in pack hook raises a RuntimeError.
    Expectation: Raise RuntimeError.
    """

    def inplace_mul_hook(x):
        x *= x
        return x

    def forward_fn(x, y):
        x = x + 1.0
        y = y + 1.0
        with ms.saved_tensors_hooks(inplace_mul_hook, lambda x: x):
            out = x * y
        return out

    x = ops.rand(5, 5, dtype=ms.float32)
    y = ops.rand(5, 5, dtype=ms.float32)
    with pytest.raises(RuntimeError, match="Pack hook inputs cannot be modified in-place"):
        ms.value_and_grad(forward_fn)(x, y)


def pack_to_cpu(x):
    return x.to('CPU')


def unpack_to_ascend(x):
    return x.to('Ascend')


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensor_hook_offload():
    """
    Features: Saved Tensor Hook.
    Description: Test saved tensor hook with offloading.
    Expectation: Gradients with and without offloading should match and memory usage should be lower with offloading.
    """
    _pynative_executor.clear_res()

    input_x = Tensor(np.random.rand(1024, 1024), dtype=ms.float32)
    memory_record = []

    def fn(x, is_offload):
        y = x * x

        def scope(y, is_offload):
            if is_offload:
                with ms.saved_tensors_hooks(pack_to_cpu, unpack_to_ascend):
                    z = ms.mint.functional.relu(y)
            else:
                z = ms.mint.functional.relu(y)
            return z + 1.0

        out = scope(y, is_offload=is_offload)
        ms.runtime.synchronize()
        memory_record.append(ms.runtime.memory_allocated())
        return out

    grad = ms.grad(fn, grad_position=0)(input_x, False).asnumpy()
    grad_offload = ms.grad(fn, grad_position=0)(input_x, True).asnumpy()

    assert np.allclose(grad, grad_offload, 0.00001, 0.00001)
    assert len(memory_record) == 2
    device_memory_gap = memory_record[0] - memory_record[1]
    assert device_memory_gap >= 10 ** 6


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensor_custom_function_hook_offload():
    """
    Features: Saved Tensor Hook.
    Description: Test saved tensor hook with offloading in custom function.
    Expectation: Gradients with and without offloading should match and memory usage should be lower with offloading.
    """
    _pynative_executor.clear_res()

    class CustomOp(_Function):
        @staticmethod
        def forward(ctx, x):
            x_mul = x * x
            ctx.save_for_backward(x_mul)
            return x_mul

        @staticmethod
        def backward(ctx, grad):
            x_mul = ctx.saved_tensors[0]
            return grad * x_mul

    input_x = Tensor(np.random.rand(1024, 1024), dtype=ms.float32)
    memory_record = []

    def fn(x, is_offload):
        def scope(x):
            if is_offload:
                with ms.saved_tensors_hooks(pack_to_cpu, unpack_to_ascend):
                    out = CustomOp.apply(x)
            else:
                out = CustomOp.apply(x)
            return out + 2.0

        res = scope(x)
        ms.runtime.synchronize()
        memory_record.append(ms.runtime.memory_allocated())
        return res

    grad = ms.grad(fn, grad_position=0)(input_x, False).asnumpy()
    grad_offload = ms.grad(fn, grad_position=0)(input_x, True).asnumpy()

    assert np.allclose(grad, grad_offload, 0.00001, 0.00001)
    assert len(memory_record) == 2
    device_memory_gap = memory_record[0] - memory_record[1]
    assert device_memory_gap >= 10 ** 6


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensors_hooks_nested_apply_innermost():
    """
    Features: Saved Tensor Hook.
    Description: Test nested saved tensors hooks to confirm innermost hook is applied.
    Expectation: Inner hook takes priority, gradients match expected values.
    """

    def fn(input_x, input_y):
        with ms.saved_tensors_hooks(lambda x: x * 2, lambda x: x):
            x_mul = input_x * input_x
            with ms.saved_tensors_hooks(
                    lambda x: x * 3, lambda x: x
            ):
                y_mul = input_y * input_y
        return x_mul + y_mul

    input_x = ops.rand(3, 3, dtype=ms.float32)
    input_y = ops.rand(3, 3, dtype=ms.float32)

    grad_x, grad_y = ms.grad(fn, grad_position=(0, 1))(input_x, input_y)
    expect_grad_x = input_x * 4
    expect_grad_y = input_y * 6
    assert np.allclose(grad_x.asnumpy(), expect_grad_x.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), expect_grad_y.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensors_hooks_cell_with_forward_hook():
    """
    Features: Saved Tensor Hook.
    Description: Verify that saved tensor hooks registered on the Cell are not triggered
                 during execution of forward hooks and forward pre-hooks.
    Expectation: Success.
    """

    def forward_hook(cell, cell_input, cell_output):
        return ops.relu(cell_output[0])

    def forward_pre_hook(cell, cell_input):
        return cell_input[0] * cell_input[0]

    class CustomNet(nn.Cell):
        def construct(self, x):
            return x * x

    WrapRecordHook.clear()

    net = CustomNet()
    net.register_saved_tensors_hooks(WrapRecordHook(pack_hook_stop_gradient, True),
                                    WrapRecordHook(lambda x: x, False))
    net.register_forward_hook(forward_hook)
    net.register_forward_pre_hook(forward_pre_hook)

    x = ops.rand(3, 3, dtype=ms.float32)
    GradOfFirstInput(net, sens_param=False)(x)
    assert WrapRecordHook.pack_record_list == [0, 0]
    assert WrapRecordHook.unpack_record_list == [0, 0]


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensors_hooks_with_recompute():
    """
    Features: Saved Tensor Hook.
    Description: Test saved tensors hooks with recompute.
    Expectation: Success.
    """

    pack_hook = WrapRecordHook(pack_hook_stop_gradient, True)
    unpack_hook = WrapRecordHook(lambda x: x - 2.0, False)

    class RecomputeNet(nn.Cell):

        def compute(self, x, y):
            z1 = x * y
            return ops.sigmoid(z1)

        def construct(self, x, y, is_inner):
            if is_inner:
                with ms.saved_tensors_hooks(pack_hook, unpack_hook):
                    return self.compute(x, y)
            return self.compute(x, y)

    WrapRecordHook.clear()

    net = RecomputeNet()
    net.recompute()
    net.register_saved_tensors_hooks(pack_hook, unpack_hook)
    net.set_grad()

    x = ops.rand(3, 3, dtype=ms.float32)
    y = ops.rand(3, 3, dtype=ms.float32)
    net(x, y, False)
    assert WrapRecordHook.pack_record_list == [0, 0]
    assert not WrapRecordHook.unpack_record_list

    GradOfAllInputs(net, sens_param=False)(x, y, False)
    assert WrapRecordHook.unpack_record_list == [0, 0]

    WrapRecordHook.clear()
    net = RecomputeNet()
    net.recompute()
    net.set_grad()
    net(x, y, True)
    assert not WrapRecordHook.pack_record_list
    assert not WrapRecordHook.unpack_record_list

    GradOfAllInputs(net, sens_param=False)(x, y, True)
    assert WrapRecordHook.pack_record_list == [0] * 3
    assert WrapRecordHook.unpack_record_list == [0] * 3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensors_hooks_with_high_grad():
    """
    Features: Saved Tensor Hook.
    Description: Test saved tensors hooks with high grad.
    Expectation: Success.
    """

    def forward_fn(x):
        with ms.saved_tensors_hooks(lambda x: x + 2.0, ops.stop_gradient):
            return ops.pow(x, 3)

    x = ops.randn(2, 2)
    high_grad_x = ms.grad(ms.grad(forward_fn))(x)
    assert np.allclose(high_grad_x.asnumpy(), (6 * (x + 2.0)).asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensors_hooks_ignore_wrapped_number():
    """
    Features: Saved Tensor Hook.
    Description: Test that wrapped numbers (e.g., scalar constants) do not trigger saved_tensors_hooks.
    Expectation: Saved tensor hooks are not called for wrapped numbers.
    """

    def fn(x):
        with ms.saved_tensors_hooks(err_hook, err_hook):
            out = ms.mint.mul(3, x)
        return out

    input_x = ops.rand(5, dtype=ms.float32)
    ms.grad(fn, grad_position=0)(input_x)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_saved_tensors_hooks_pre_disable_async():
    """
    Features: Saved Tensor Hook.
    Description: Verify the behavior of pre_disable_async when entering nested saved tensor hooks.
    Expectation: The first context keeps pre_disable_async as False, while the nested context sets it to True.
    """
    ctx1 = ms.saved_tensors_hooks(pack_hook_stop_gradient, pack_hook_stop_gradient)
    ctx2 = ms.saved_tensors_hooks(pack_hook_stop_gradient, pack_hook_stop_gradient)
    assert not ctx1.pre_disable_async

    ctx1.__enter__()
    assert not ctx1.pre_disable_async
    ctx2.__enter__()
    assert ctx2.pre_disable_async

    ctx2.__exit__()
    ctx1.__exit__()

    ctx3 = ms.saved_tensors_hooks(pack_hook_stop_gradient, pack_hook_stop_gradient)
    ctx3.__enter__()
    assert not ctx3.pre_disable_async
    ctx3.__exit__()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_disable_saved_tensor_hook():
    """
    Features: Disable Saved Tensor Hook.
    Description: Verify that disable_saved_tensors_hooks works correctly in saved tensor hook context.
    Expectation: The correct error message is raised when hooks are disabled, and gradients match the expected values.
    """
    with _disable_saved_tensors_hooks("error message"):
        with pytest.raises(RuntimeError, match="error message"):
            with ms.saved_tensors_hooks(lambda x: x, lambda x: x):
                pass

    with ms.saved_tensors_hooks(lambda x: x, lambda x: x):
        with pytest.raises(RuntimeError, match="error message"):
            with _disable_saved_tensors_hooks("error message"):
                pass

    def fn(input_x):
        with ms.saved_tensors_hooks(lambda x: x + 1.0, lambda x: x + 1.0):
            y = input_x * input_x
            with _disable_saved_tensors_hooks("error message", is_error_on_outer_hook=False):
                out = y * y
        return out

    input_tensor = ops.rand(5, dtype=ms.float32)
    grad = ms.grad(fn, grad_position=0)(input_tensor)
    expect_grad = 2 * (input_tensor * input_tensor) * (input_tensor + 2.0) * 2
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_disable_saved_tensor_hook_nested():
    """
    Features: Disable Saved Tensor Hook.
    Description: Verify that nested disable_saved_tensors_hooks prioritizes the inner error message.
    Expectation: The inner error message is raised.
    """
    with _disable_saved_tensors_hooks("outer"):
        with _disable_saved_tensors_hooks("inner"):
            with pytest.raises(RuntimeError, match="inner"):
                with ms.saved_tensors_hooks(
                        lambda x: x + 1.0, lambda x: x * 2.0
                ):
                    pass
