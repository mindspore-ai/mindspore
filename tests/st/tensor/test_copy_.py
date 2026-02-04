# Copyright 2024 Huawei Technologies Co., Ltd
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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, x, y):
        z = x * 1
        return z.copy_(y)


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def copy_forward_func(x, y):
    return Net()(x, y)


def copy_backward_func(x, y):
    return ops.grad(copy_forward_func, grad_position=(0, 1))(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_copy_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 2, 3, 4), np.float32)
    y = generate_random_input((2, 2, 3, 4), np.float32)
    z = generate_random_input((2, 1, 4), np.float32)  # broadcast

    expect_y_grad = np.ones_like(y, dtype=np.float32)
    expect_z = np.expand_dims(z.repeat(3, axis=1), axis=0).repeat(2, axis=0)
    expect_z_grad = np.ones_like(z, dtype=np.float32) * 6

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_y = copy_forward_func(ms.Tensor(x), ms.Tensor(y))
        output_y_grad = copy_backward_func(ms.Tensor(x), ms.Tensor(y))

        output_z = copy_forward_func(ms.Tensor(x), ms.Tensor(z))
        output_z_grad = copy_backward_func(ms.Tensor(x), ms.Tensor(z))
    else:
        output_y = (jit(copy_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(y))
        output_y_grad = (jit(copy_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(y))

        output_z = (jit(copy_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(z))
        output_z_grad = (jit(copy_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(z))
    np.allclose(output_y.asnumpy(), y, rtol=1e-5, equal_nan=True)
    np.allclose(output_y_grad[1].asnumpy(), expect_y_grad, rtol=1e-5, equal_nan=True)

    np.allclose(output_z.asnumpy(), expect_z, rtol=1e-5, equal_nan=True)
    np.allclose(output_z_grad[1].asnumpy(), expect_z_grad, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_copy_cpu_inplace_feature():
    """
    Feature: inplace features.
    Description: test copy forward .
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    expect = np.array([0, 6, 5, 6]).astype(np.int64).reshape((2, 2))
    inputs = [
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([[5, 6], [5, 6]])],
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([[5., 6.], [5., 6.]])],
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([5, 6])],
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([5., 6.])],
    ]
    for param in inputs:
        x = param[0]
        y = param[1]
        z = x.copy_(y)
        x[0][0] = 0
        np.testing.assert_equal(expect, x.asnumpy())
        np.testing.assert_equal(expect, z.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_copy_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test copy forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_y1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y2 = ms.Tensor(generate_random_input((1, 1, 5), np.float32))  # broadcast

    TEST_OP(copy_forward_func, [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'all_dim_zero': True})
    TEST_OP(copy_forward_func, [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
def test_copy_bfloat16():
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 2, 5, 4), np.float32)
    y = generate_random_input((2, 1, 4), np.float32)

    expect = np.expand_dims(y.repeat(5, axis=1), axis=0).repeat(3, axis=0)
    expect_grad = np.ones_like(y).astype(np.float32) * 15

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = copy_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))
    output_grad = copy_backward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect, 0.004, 0.004, equal_nan=True)
    np.allclose(output_grad[1].float().asnumpy(), expect_grad, 0.004, 0.004, equal_nan=True)


def copy_h2d_d2h_h2h(non_blocking):
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API with h2d, d2h and h2h.
    Expectation: the result match with expected result.
    """
    #h2d
    ori_alloc_mem = ms.runtime.memory_allocated()
    dst1 = ms.mint.randn((1024, 1024))
    alloc_mem = ms.runtime.memory_allocated()
    src1 = ms.Tensor(np.random.randn(1024, 1024).astype(np.float32))
    dst1.copy_(src1, non_blocking=non_blocking)
    assert alloc_mem == ms.runtime.memory_allocated()
    assert np.all(dst1.asnumpy() == src1.asnumpy())
    dst1.storage().resize_(0)
    assert ori_alloc_mem == ms.runtime.memory_allocated()

    #d2h
    ori_alloc_mem = ms.runtime.memory_allocated()
    src2 = ms.mint.randn((1024, 1024))
    alloc_mem = ms.runtime.memory_allocated()
    dst2 = ms.mint.empty_like(src2, device="CPU")
    dst2.copy_(src2, non_blocking=non_blocking)
    assert alloc_mem == ms.runtime.memory_allocated()
    assert np.all(dst2.asnumpy() == src2.asnumpy())
    src2.storage().resize_(0)
    assert ori_alloc_mem == ms.runtime.memory_allocated()

    #h2h
    alloc_mem = ms.runtime.memory_allocated()
    src3 = ms.Tensor(np.random.randn(1024, 1024))
    dst3 = ms.Tensor(np.random.randn(1024, 1024))
    dst3.copy_(src3, non_blocking=non_blocking)
    assert alloc_mem == ms.runtime.memory_allocated()
    assert np.all(dst3.asnumpy() == src3.asnumpy())


def copy_h2d_d2h_view(non_blocking):
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API with h2d and d2h view.
    Expectation: the result match with expected result.
    """
    #h2d
    dst1 = ms.mint.randn((1024, 1024))
    alloc_mem = ms.runtime.memory_allocated()
    view1 = dst1[1]
    src1 = ms.Tensor(np.random.randn(1024,).astype(np.float32))
    view1.copy_(src1, non_blocking=non_blocking)
    assert alloc_mem == ms.runtime.memory_allocated()
    assert np.all(view1.asnumpy() == src1.asnumpy())

    #d2h
    src2 = ms.mint.randn((1024, 1024))
    alloc_mem = ms.runtime.memory_allocated()
    view2 = src2[1]
    dst2 = ms.mint.empty_like(view2, device="CPU")
    dst2.copy_(view2, non_blocking=non_blocking)
    assert alloc_mem == ms.runtime.memory_allocated()
    assert np.all(dst2.asnumpy() == view2.asnumpy())


def copy_h2d_d2h_discontiguous(non_blocking):
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API with h2d and d2h discontiguous.
    Expectation: the result match with expected result.
    """
    #h2d
    alloc_mem1 = ms.runtime.memory_allocated()
    dst1 = ms.mint.randn((512, 1024), dtype=ms.float32)
    discontig1 = dst1.t()
    src1 = ms.Tensor(np.random.randn(1024, 512).astype(np.float32))
    discontig1.copy_(src1, non_blocking=non_blocking)
    assert (ms.runtime.memory_allocated() - alloc_mem1) == 4195328
    assert np.all(discontig1.asnumpy() == src1.asnumpy())

    #d2h
    alloc_mem2 = ms.runtime.memory_allocated()
    src2 = ms.mint.randn((512, 1024), dtype=ms.float32)
    discontig2 = src2.t()
    dst2 = ms.mint.empty_like(discontig2, device="CPU")
    dst2.copy_(discontig2, non_blocking=non_blocking)
    assert (ms.runtime.memory_allocated() - alloc_mem2) == 4195328
    assert np.all(dst2.asnumpy() == discontig2.asnumpy())


def copy_h2d_d2h_h2h_empty(non_blocking):
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API with h2d and d2h empty src/dst.
    Expectation: the result match with expected result.
    """
    x = ms.ops.slice(ms.Tensor([1], dtype=ms.float32), (0,), (0,))
    alloc_mem1 = ms.runtime.memory_allocated()
    y = ms.mint.empty_like(x, device="CPU")
    x.copy_(y, non_blocking=non_blocking)
    assert alloc_mem1 == ms.runtime.memory_allocated()
    assert np.all(x.asnumpy() == y.asnumpy())

    alloc_mem2 = ms.runtime.memory_allocated()
    z = ms.mint.empty_like(x, device="CPU")
    z.copy_(x, non_blocking=non_blocking)
    assert alloc_mem2 == ms.runtime.memory_allocated()
    assert np.all(x.asnumpy() == z.asnumpy())

    alloc_mem3 = ms.runtime.memory_allocated()
    a = ms.mint.empty_like(x, device="CPU")
    b = ms.mint.empty_like(x, device="CPU")
    a.copy_(b, non_blocking=non_blocking)
    assert alloc_mem3 == ms.runtime.memory_allocated()
    assert np.all(a.asnumpy() == b.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('non_blocking', [False, True])
def test_copy_h2d_d2h_h2h(non_blocking):
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API with h2d, d2h and h2h with view and discontiguous.
    Expectation: the result match with expected result.
    """
    copy_h2d_d2h_h2h(non_blocking)
    copy_h2d_d2h_view(non_blocking)
    copy_h2d_d2h_discontiguous(non_blocking)
    copy_h2d_d2h_h2h_empty(non_blocking)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_copy_large_tensor():
    """
    Feature: test copy tensor API.
    Description: testcase for copy tensor API with large tensor on cpu.
    Expectation: the result match with expected result.
    """
    src1 = ms.Tensor(np.random.randn(380762112, 2), dtype=ms.float32)
    dst1 = ms.mint.empty_like(src1, device="CPU")
    dst1.copy_(src1)
    assert np.all(dst1.asnumpy() == src1.asnumpy())

    src2 = ms.Tensor(np.random.randn(380762112, 2), dtype=ms.float32)
    dst2 = ms.mint.empty((2, 380762112, 2), dtype=src2.dtype, device="CPU")
    dst2.copy_(src2)
    assert np.all(dst2.asnumpy() == np.broadcast_to(src2.asnumpy(), (2, 380762112, 2)))
