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
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils.test_utils import get_inputs_np, get_inputs_tensor

import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, mint
from mindspore.common import dtype as mstype
import pytest


# pylint: disable=W0235


def _ms_access_real_imag(x, accessor: str):
    if accessor == "func":
        return mint.real(x), mint.imag(x)
    return x.real(), x.imag()


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """Compare if two numpy arrays are equal within tolerance"""
    if not np.allclose(data_expected, data_me, rtol, rtol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """Count the number of unequal elements"""
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize("accessor", ["func", "attr"])  # func: mint.real/mint.imag, attr: x.real/x.imag
def test_real_imag_compare_numpy(accessor):
    """
    Feature: Real/Imag ascend kernel compare with numpy (function and tensor property)
    Description: Compare the real and imag view results between MindSpore and NumPy via two access paths
    Expectation: outputs are the same as numpy within tolerance
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    
    # 固定随机种子
    np.random.seed(42)

    def run_unified_test(real_data, imag_data, dtype_ms, dtype_np, rtol, atol, _name):
        # MindSpore input and output
        real_ms = Tensor(real_data, dtype_ms)
        imag_ms = Tensor(imag_data, dtype_ms)
        x_ms = ops.Complex()(real_ms, imag_ms)
        ms_real, ms_imag = _ms_access_real_imag(x_ms, accessor)
        ms_real = ms_real.asnumpy()
        ms_imag = ms_imag.asnumpy()

        # NumPy input and output
        numpy_real, numpy_imag = real_data, imag_data

        # NaN consistency
        if np.any(np.isnan(real_data)) or np.any(np.isnan(imag_data)):
            assert np.array_equal(np.isnan(ms_real), np.isnan(numpy_real))
            assert np.array_equal(np.isnan(ms_imag), np.isnan(numpy_imag))

        # Inf consistency
        if np.any(np.isinf(real_data)) or np.any(np.isinf(imag_data)):
            assert np.array_equal(np.isinf(ms_real), np.isinf(numpy_real))
            assert np.array_equal(np.isinf(ms_imag), np.isinf(numpy_imag))

        # Finite value comparison
        finite_mask_real = np.isfinite(ms_real)
        finite_mask_imag = np.isfinite(ms_imag)
        if np.any(finite_mask_real):
            allclose_nparray(numpy_real[finite_mask_real], ms_real[finite_mask_real], rtol, atol)
        if np.any(finite_mask_imag):
            allclose_nparray(numpy_imag[finite_mask_imag], ms_imag[finite_mask_imag], rtol, atol)

    # float32 complex numbers
    real_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float32)
    imag_data = np.array([7.7, 8.8, 9.9, 10.0, 11.1, 12.2], dtype=np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, np.float32, 1e-5, 1e-5, "float32 complex")

    # float64 complex numbers
    real_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)
    imag_data = np.array([7.7, 8.8, 9.9, 10.0, 11.1, 12.2], dtype=np.float64)
    run_unified_test(real_data, imag_data, mstype.float64, np.float64, 1e-10, 1e-10, "float64 complex")

    # Random data
    batch_size, channels, height, width = 2, 4, 8, 8
    real_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    imag_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, np.float32, 1e-5, 1e-5, "random data")

    # Zero values
    real_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imag_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, np.float32, 1e-5, 1e-5, "zero values")

    # NaN values
    real_data = np.array([np.nan, 1.0, np.nan], dtype=np.float32)
    imag_data = np.array([2.0, np.nan, np.nan], dtype=np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, np.float32, 1e-5, 1e-5, "NaN values")

    # Infinity values
    real_data = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
    imag_data = np.array([2.0, np.inf, -np.inf], dtype=np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, np.float32, 1e-5, 1e-5, "infinity values")

    # Mixed special values
    real_data = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=np.float32)
    imag_data = np.array([np.inf, np.nan, 0.0, -np.inf, 2.0], dtype=np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, np.float32, 1e-5, 1e-5, "mixed special values")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_real_imag_view_exceptions():
    """
    Feature: RealImagView ascend kernel exception handling
    Description: Test exception scenarios including TypeError and ValueError
    Expectation: appropriate exceptions are raised for invalid inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    def test_exception(exception_type, data=None, dtype=None, operation_name=None, test_type="type_error"):
        """Helper function for testing exceptions, supports TypeError and ValueError"""
        with pytest.raises(exception_type) as exc_info:
            view_op = mint.real if operation_name == "real_view" else mint.imag
            # TypeError test: non-complex type input
            input_tensor = Tensor(data, dtype)
            view_op(input_tensor)
            ms.runtime.synchronize()

        assert "complex64 or complex128" in str(exc_info.value)

    # Test case 1: TypeError - non-complex type input
    # Test float32 type
    test_exception(TypeError, [1.0, 2.0, 3.0], mstype.float32, "imag_view", "type_error")

    # Test float64 type
    test_exception(TypeError, [1.0, 2.0, 3.0], mstype.float64, "imag_view", "type_error")

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_real_imag_view_backward():
    """
    Feature: RealImagView ascend kernel backward propagation
    Description: Test gradient computation and backward propagation for real and imag view operations
    Expectation: gradients are computed correctly
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    
    # 固定随机种子
    np.random.seed(42)

    # Create complex input data
    batch_size, channels, height, width = 2, 3, 4, 4
    real_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    imag_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # MindSpore network
    class MSBackwardNet(nn.Cell):
        def __init__(self):
            super(MSBackwardNet, self).__init__()
            self.real_view = mint.real
            self.imag_view = mint.imag
            self.sum = ops.ReduceSum()

        def construct(self, complex_input):
            # Separate real and imaginary parts
            real_part = self.real_view(complex_input)
            imag_part = self.imag_view(complex_input)

            # Calculate loss (sum of real and imaginary parts)
            loss = self.sum(real_part) + self.sum(imag_part)
            return loss

    # Initialize network
    ms_net = MSBackwardNet()

    # MindSpore input
    real_ms = Tensor(real_data, mstype.float32)
    imag_ms = Tensor(imag_data, mstype.float32)
    complex_input_ms = ops.Complex()(real_ms, imag_ms)
    complex_input_ms.requires_grad = True

    # Get mindspore grad
    ms_grad_fn = ms.value_and_grad(ms_net, grad_position=0)
    ms_loss, ms_grad = ms_grad_fn(complex_input_ms)

    # Verify that gradients are computed
    assert ms_grad is not None
    assert ms_grad.shape == complex_input_ms.shape
    assert ms_grad.dtype == complex_input_ms.dtype


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_real_view_non_complex_input_no_error():
    """
    Feature: RealView ascend kernel with non-complex input
    Description: real on non-complex tensor should not raise and keep dtype unchanged
    Expectation: no exception, output equals input and dtype unchanged
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    for dt in (mstype.float16, mstype.float32, mstype.float64):
        data = np.array([1.25, -2.5, 3.0], dtype={
            mstype.float16: np.float16,
            mstype.float32: np.float32,
            mstype.float64: np.float64,
        }[dt])
        x = Tensor(data, dt)
        y = mint.real(x)
        assert y.dtype == dt
        np.testing.assert_allclose(y.asnumpy(), data, rtol=1e-5, atol=1e-5)


def real_func(x):
    return mint.real(x)

def imag_func(x):
    return mint.imag(x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
def test_real_imag_view_dyn():
    """
    Feature: RealView and ImagView ascend kernel with dynamic shape
    Description: real and imag view should support dynamic shape
    Expectation: no exception, output equals input and dtype unchanged
    """
    np.random.seed(42)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    input_case1 = get_inputs_tensor(get_inputs_np([(2,4,4)], [np.complex64]))
    input_case2 = get_inputs_tensor(get_inputs_np([(2,2,4,4)], [np.complex64]))
    TEST_OP(real_func, [input_case1, input_case2], disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])
    TEST_OP(imag_func, [input_case1, input_case2], disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])
