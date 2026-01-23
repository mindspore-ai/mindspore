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
from mindspore.device_context.cpu.op_tuning import threads_num
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase

def concat_func(x1, x2, axis):
    return F.concat((x1, x2), axis=axis)


def concat_bwd_func(x1, x2, axis):
    return ms.grad(concat_func, (0, 1))(x1, x2, axis)


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


def forward_datas_prepare(shape, num=2, axis=0, diff_shapes=False, need_expect=True, numpy_inputs=False,\
                          is_bfloat16=False):
    np_inpus = []
    tensor_inputs = []
    if diff_shapes:
        if not isinstance(shape, (list, tuple)):
            raise RuntimeError("shape should be list or tuple, but got %s(type %s)." % (shape, type(shape)))
        num = len(shape)
    for i in range(num):
        np_input = np.random.rand(*(shape[i] if diff_shapes else shape)).astype(np.float32)
        np_inpus.append(np_input)
        t = ms.Tensor(np_input, ms.bfloat16) if is_bfloat16 else ms.Tensor(np_input)
        tensor_inputs.append(t)
    np_expect = np.concatenate(np_inpus, axis) if need_expect else None
    return tuple(np_inpus if numpy_inputs else tensor_inputs), np_expect


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_concat_normal(mode, params):
    """
    Feature: Ops.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_level='O0')
    shape_param, axis = params
    tensor_inputs, expect = forward_datas_prepare(shape_param, axis=axis, diff_shapes=True)
    out = concat_forward_func(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.asnumpy(), expect)

    x1 = ms.Tensor(np.random.rand(*shape_param[0]).astype(np.float32))
    x2 = ms.Tensor(np.random.rand(*shape_param[1]).astype(np.float32))
    grads = concat_backward_func(x1, x2, axis)
    expect_grad1 = np.ones(shape_param[0]).astype(np.float32)
    expect_grad2 = np.ones(shape_param[1]).astype(np.float32)
    expect_grad = (expect_grad1, expect_grad2)
    for out, expect in zip(grads, expect_grad):
        assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_concat_bfloat16(mode, params):
    """
    Feature: Ops.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_level='O0')
    shape_param, axis = params
    tensor_inputs, expect = forward_datas_prepare(shape_param, axis=axis, diff_shapes=True, is_bfloat16=True)
    out = concat_forward_func(tensor_inputs[0], tensor_inputs[1], axis)
    assert np.allclose(out.float().asnumpy(), expect, 4e-3, 4e-3)

    x1 = ms.Tensor(np.random.rand(*shape_param[0]).astype(np.float32), ms.bfloat16)
    x2 = ms.Tensor(np.random.rand(*shape_param[1]).astype(np.float32), ms.bfloat16)
    grads = concat_backward_func(x1, x2, axis)
    expect_grad1 = np.ones(shape_param[0]).astype(np.float32)
    expect_grad2 = np.ones(shape_param[1]).astype(np.float32)
    expect_grad = (expect_grad1, expect_grad2)
    for out, expect in zip(grads, expect_grad):
        assert np.allclose(out.float().asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_concat_mixed_dtype(mode):
    """
    Feature: Ops.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_level='O0')
    shape_param = ((2, 2), (2, 3))
    axis = 1
    x1_np = np.random.rand(*shape_param[0]).astype(np.float32)
    x2_np = np.random.rand(*shape_param[1]).astype(np.float32)
    x1 = ms.Tensor(x1_np, ms.float16)
    x2 = ms.Tensor(x2_np, ms.float32)
    out = concat_forward_func(x1, x2, axis)
    np_expect = np.concatenate((x1_np, x2_np), axis)
    assert np.allclose(out.asnumpy(), np_expect, 1e-3, 1e-3)
    grads = concat_backward_func(x1, x2, axis)
    expect_grad1 = np.ones(x1_np.shape).astype(np.float16)
    expect_grad2 = np.ones(x2_np.shape).astype(np.float32)
    expect_grad = (expect_grad1, expect_grad2)
    for out, expect in zip(grads, expect_grad):
        assert np.allclose(out.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
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
          card_mark='onecard', essential_mark='essential', mem_peak=3072.00)
def test_concat_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op concat.
    Expectation: expect correct result.
    """
    threads_num(1)
    axis = 1
    inputs_case1, _ = forward_datas_prepare((2, 4), axis=axis, need_expect=False)
    inputs_case2, _ = forward_datas_prepare((2, 2, 2), axis=axis, need_expect=False)
    TEST_OP(concat_func, [[*inputs_case1, axis], [*inputs_case2, axis]],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_forward_dynamic(mode, dyn_mode):
    """
    Feature: test dynamic.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_level='O0')
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
          card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_backward_dynamic(mode, dyn_mode):
    """
    Feature: test dynamic.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_level='O0')
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
    ms.set_context(mode=mode, jit_level='O0')
    axis = 1
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    fwd_seq_cell = test_utils.to_cell_obj(concat_dyn_seq_fwd_func)
    fwd_seq_cell.set_inputs(ms.mutable((x1_dyn, x2_dyn), True), axis)

    shape1 = (2, 4)
    num1 = 2
    tensor_inputs1, expect1 = forward_datas_prepare(shape1, num=num1, axis=axis)
    out1 = fwd_seq_cell(ms.mutable(tensor_inputs1), axis)
    assert np.allclose(out1.asnumpy(), expect1)

    # Dynamic sequence only support same shape inner now.
    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    num2 = 2  # Should be different, set 2 here for mutable limit.
    tensor_inputs2, expect2 = forward_datas_prepare(shape2, num=num2, axis=axis)
    out2 = fwd_seq_cell(ms.mutable(tensor_inputs2), axis)
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
    grads1 = bwd_seq_cell(ms.mutable(input_seq1), axis)
    expect_grad1 = (np.ones(shape1).astype(np.float32),) * num1
    for out, expect in zip(grads1, expect_grad1):
        assert np.allclose(out.asnumpy(), expect)

    # Dynamic sequence only support same shape inner now.
    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    num2 = 2  # Should be different, set 2 here for mutable bug.
    input_seq2, _ = forward_datas_prepare(shape2, num=num2, axis=axis, need_expect=False)
    grad2 = bwd_seq_cell(ms.mutable(input_seq2), axis)
    expect_grad2 = (np.ones(shape2).astype(np.float32),) * num2
    for out, expect in zip(grad2, expect_grad2):
        assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_concat_with_single_input():
    """
    Feature: Concat ops.
    Description: Test Concat with single input.
    Expectation: No Exception raised.
    """
    class Net(nn.Cell):
        def construct(self, x):
            x = ops.add(x, x)
            y = ops.concat([x,])
            return y

    context.set_context(mode=context.GRAPH_MODE, jit_level="O0")
    net = Net()
    x = Tensor(np.random.rand(100, 10).astype(np.float32))
    out = net(x)
    print(out)


def ops_concat_binary_compare(input_binary_data, output_binary_data, dim, is_bf16=False):

    @test_utils.run_with_cell
    def cat_forward_func(x1, x2, dim):
        return ms.mint.cat((x1, x2), dim)


    @test_utils.run_with_cell
    def cat_backward_func(x1, x2, dim):
        return ms.grad(cat_forward_func, (0, 1))(x1, x2, dim)

    if is_bf16:
        tensors = [ms.Tensor(np_data, ms.bfloat16) for np_data in input_binary_data]
        loss = 4e-3
    else:
        tensors = [ms.Tensor(np_data) for np_data in input_binary_data]
        loss = 1e-4
    output = cat_forward_func(tensors[0], tensors[1], dim)
    if is_bf16:
        assert np.allclose(output.float().asnumpy(), output_binary_data[0], loss, loss)
    else:
        assert np.allclose(output.asnumpy(), output_binary_data[0], loss, loss)

    grads = cat_backward_func(tensors[0], tensors[1], dim)
    for i, expect in enumerate(output_binary_data[1:]):
        if is_bf16:
            assert np.allclose(grads[i].float().asnumpy(), expect, loss, loss)
        else:
            assert np.allclose(grads[i].asnumpy(), expect, loss, loss)


@ops_binary_cases(OpsBinaryCase(input_info=[((4608, 32), np.float32), ((4608, 32), np.float32)],
                                output_info=[((4608, 64), np.float32), ((4608, 32), np.float32),
                                             ((4608, 32), np.float32)],
                                extra_info='auto_drive'))
def ops_concat_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_concat_binary_compare(input_binary_data, output_binary_data, 1)


@ops_binary_cases(OpsBinaryCase(input_info=[((2048, 5120), np.float32), ((2048, 5120), np.float32)],
                                output_info=[((4096, 5120), np.float32), ((2048, 5120), np.float32),
                                             ((2048, 5120), np.float32)],
                                extra_info='pg135B'))
def ops_concat_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_concat_binary_compare(input_binary_data, output_binary_data, 0, True)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 1, 5120), np.float32), ((1, 4096, 5120), np.float32)],
                                output_info=[((1, 4097, 5120), np.float32), ((1, 1, 5120), np.float32),
                                             ((1, 4096, 5120), np.float32)],
                                extra_info='pgmoe'))
def ops_concat_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_concat_binary_compare(input_binary_data, output_binary_data, 1, True)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 8, 1, 5120), np.float32), ((1, 8, 1544, 5120), np.float32)],
                                output_info=[((1, 8, 1545, 5120), np.float32), ((1, 8, 1, 5120), np.float32),
                                             ((1, 8, 1544, 5120), np.float32)],
                                extra_info='pgmoe'))
def ops_concat_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_concat_binary_compare(input_binary_data, output_binary_data, 2, True)


@ops_binary_cases(OpsBinaryCase(input_info=[((2048, 6144), np.float32), ((2048, 6144), np.float32)],
                                output_info=[((4096, 6144), np.float32), ((2048, 6144), np.float32),
                                             ((2048, 6144), np.float32)],
                                extra_info='pg230b'))
def ops_concat_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_concat_binary_compare(input_binary_data, output_binary_data, 0, True)


@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_concat_binary_cases(mode):
    """
    Feature: Ops.
    Description: test op concat with binary data.
    Expectation: expect correct result.
    """
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    ops_concat_binary_case1()
    ops_concat_binary_case2()
    ops_concat_binary_case3()
    ops_concat_binary_case4()
    ops_concat_binary_case5()
