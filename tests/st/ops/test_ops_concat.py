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

""" Test concat. """

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, nn, Tensor, context, mutable
import mindspore.ops.functional as F
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def concat_func(x1, x2, axis):
    return F.concat((x1, x2), axis=axis)


def concat_bwd_func(x1, x2, axis):
    return ops.grad(concat_func, (0, 1))(x1, x2, axis)


@test_utils.run_with_cell
def concat_forward_func(x1, x2, axis):
    return concat_func(x1, x2, axis)


@test_utils.run_with_cell
def concat_backward_func(x1, x2, axis):
    return concat_bwd_func(x1, x2, axis)


def concat_dyn_seq_fwd_func(seq, axis):
    return F.concat(seq, axis=axis)


def concat_dyn_seq_bwd_func(seq, axis):
    return ops.grad(concat_dyn_seq_fwd_func, (0,))(seq, axis)


def forward_datas_prepare(shape, num=2, axis=0, diff_shapes=False, need_expect=True, numpy_inputs=False):
    np_inpus = []
    tensor_inputs = []
    if diff_shapes:
        if not isinstance(shape, (list, tuple)):
            raise RuntimeError("shape should be list or tuple, but got %s(type %s)." % (shape, type(shape)))
        num = len(shape)
    for i in range(num):
        np_input = np.random.rand(*(shape[i] if diff_shapes else shape)).astype(np.float32)
        np_inpus.append(np_input)
        tensor_inputs.append(ms.Tensor(np_input))
    np_expect = np.concatenate(np_inpus, axis) if need_expect else None
    return tuple(np_inpus if numpy_inputs else tensor_inputs), np_expect


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_concat_forward(mode, params):
    """
    Feature: Ops.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    shape_param, axis = params
    tensor_inputs, expect = forward_datas_prepare(shape_param, axis=axis, diff_shapes=True)
    out = concat_forward_func(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_concat_backward(mode, params):
    """
    Feature: Auto grad.
    Description: test auto grad of op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    shape_param, axis = params
    x1 = ms.Tensor(np.random.rand(*shape_param[0]).astype(np.float32))
    x2 = ms.Tensor(np.random.rand(*shape_param[1]).astype(np.float32))
    grads = concat_backward_func(x1, x2, axis)
    expect_grad1 = np.ones(shape_param[0]).astype(np.float32)
    expect_grad2 = np.ones(shape_param[1]).astype(np.float32)
    expect_grad = (expect_grad1, expect_grad2)
    for out, expect in zip(grads, expect_grad):
        assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_concat_vmap(mode, params):
    """
    Feature: test vmap function.
    Description: test concat op vmap.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    shape_param, axis = params
    in_axes = (-1, -1, None)
    tensor_np_inputs, expect_np = forward_datas_prepare(shape_param, axis=axis, diff_shapes=True, numpy_inputs=True)
    x1_np, x2_np = tensor_np_inputs
    x1 = ms.Tensor(np.tile(x1_np.reshape((shape_param[0]) + (1, 1)), (1,) * len(shape_param[0]) + (2, 2)))
    x2 = ms.Tensor(np.tile(x2_np.reshape((shape_param[1]) + (1, 1)), (1,) * len(shape_param[1]) + (2, 2)))
    nest_vmap = ops.vmap(ops.vmap(concat_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x1, x2, axis)
    expect = np.tile(expect_np.reshape((1, 1) + expect_np.shape), (2, 2) + (1,) * expect_np.ndim)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_concat_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.
    axis = 1
    inputs_case1, _ = forward_datas_prepare((2, 4), axis=axis, need_expect=False)
    inputs_case2, _ = forward_datas_prepare((2, 2, 2), axis=axis, need_expect=False)
    TEST_OP(concat_func, [[*inputs_case1, axis], [*inputs_case2, axis]], '', disable_input_check=True,
            disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_forward_dynamic(mode, dyn_mode):
    """
    Feature: test dynamic.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    axis = 1
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    fwd_cell = test_utils.to_cell_obj(concat_func)
    fwd_cell.set_inputs(x1_dyn, x2_dyn, axis)

    shape1 = (2, 4)
    tensor_inputs1, expect1 = forward_datas_prepare(shape1, axis=axis)
    out1 = fwd_cell(*tensor_inputs1, axis)
    assert np.allclose(out1.asnumpy(), expect1)

    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    tensor_inputs2, expect2 = forward_datas_prepare(shape2, axis=axis)
    out2 = fwd_cell(*tensor_inputs2, axis)
    assert np.allclose(out2.asnumpy(), expect2)

    shapes3 = ((3, 3), (3, 2)) if dyn_mode == "dyn_shape" else ((2, 2, 2), (2, 3, 2))
    tensor_inputs3, expect3 = forward_datas_prepare(shapes3, axis=axis, diff_shapes=True)
    out3 = fwd_cell(*tensor_inputs3, axis)
    assert np.allclose(out3.asnumpy(), expect3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_backward_dynamic(mode, dyn_mode):
    """
    Feature: test dynamic.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    axis = 1
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    bwd_cell = test_utils.to_cell_obj(concat_bwd_func)
    bwd_cell.set_inputs(x1_dyn, x2_dyn, axis)

    shape1 = (2, 4)
    (x1_1, x1_2), _ = forward_datas_prepare(shape1, axis=axis, need_expect=False)
    grads1 = bwd_cell(x1_1, x1_2, axis)
    expect_grad1 = (np.ones(shape1).astype(np.float32),) * 2
    for out, expect in zip(grads1, expect_grad1):
        assert np.allclose(out.asnumpy(), expect)

    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    (x2_1, x2_2), _ = forward_datas_prepare(shape2, axis=axis, need_expect=False)
    grads2 = bwd_cell(x2_1, x2_2, axis)
    expect_grad2 = (np.ones(shape2).astype(np.float32),) * 2
    for out, expect in zip(grads2, expect_grad2):
        assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_forward_dyn_seq(mode, dyn_mode):
    """
    Feature: test forward dynamic sequence.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    axis = 1
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    fwd_seq_cell = test_utils.to_cell_obj(concat_dyn_seq_fwd_func)
    fwd_seq_cell.set_inputs(ms.mutable((x1_dyn, x2_dyn), True), axis)

    shape1 = (2, 4)
    num1 = 2
    tensor_inputs1, expect1 = forward_datas_prepare(shape1, num=num1, axis=axis)
    out1 = fwd_seq_cell(tensor_inputs1, axis)
    assert np.allclose(out1.asnumpy(), expect1)

    # Dynamic sequence only support same shape inner now.
    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    num2 = 2  # Should be different, set 2 here for mutable limit.
    tensor_inputs2, expect2 = forward_datas_prepare(shape2, num=num2, axis=axis)
    out2 = fwd_seq_cell(tensor_inputs2, axis)
    assert np.allclose(out2.asnumpy(), expect2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_backward_dyn_seq(mode, dyn_mode):
    """
    Feature: test backward dynamic sequence.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    axis = 1
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    bwd_seq_cell = test_utils.to_cell_obj(concat_dyn_seq_bwd_func)
    bwd_seq_cell.set_inputs(ms.mutable((x1_dyn, x2_dyn), True), axis)

    shape1 = (2, 4)
    num1 = 2
    input_seq1, _ = forward_datas_prepare(shape1, num=num1, axis=axis, need_expect=False)
    grads1 = bwd_seq_cell(input_seq1, axis)
    expect_grad1 = (np.ones(shape1).astype(np.float32),) * num1
    for out, expect in zip(grads1, expect_grad1):
        assert np.allclose(out.asnumpy(), expect)

    # Dynamic sequence only support same shape inner now.
    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    num2 = 2  # Should be different, set 2 here for mutable bug.
    input_seq2, _ = forward_datas_prepare(shape2, num=num2, axis=axis, need_expect=False)
    grad2 = bwd_seq_cell(input_seq2, axis)
    expect_grad2 = (np.ones(shape2).astype(np.float32),) * num2
    for out, expect in zip(grad2, expect_grad2):
        assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_concat_with_input_complex64(mode):
    """
    Feature: Test Concat with input of complex64 type.
    Description: Test Concat with input of complex64 type.
    Expectation: Expect correct shape result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.concat = ops.Concat()

        def construct(self, *inputs):
            return self.concat(inputs)

    ms.set_context(mode=mode)

    input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.complex64))
    input_x2 = Tensor(np.array([[3, 2], [2, 5]]).astype(np.complex64))

    net = Net()
    out = net(input_x1, input_x2)
    expect_out = np.array([[0, 1], [2, 1], [3, 2], [2, 5]]).astype(np.complex64)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_concat_with_dyn_len_sequence_input():
    """
    Feature: Dynamic shape.
    Description: Test Concat with dyn len sequence input.
    Expectation: No Exception raised.
    """
    class Grad(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.gn = ops.GradOperation()(net)

        def construct(self, x):
            g = self.gn(x)
            return g

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.concat(x)
            return y

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    x = (Tensor([1]), Tensor([2]), Tensor([3]))
    y = mutable(x, dynamic_len=True)
    grad_net = Grad(net)
    grad = grad_net(y)
    print(grad)
