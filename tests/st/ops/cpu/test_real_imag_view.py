# Copyright 2026 Huawei Technologies Co., Ltd
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
"""CPU PyNative tests for native real and imag views."""

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_real_imag_noncontiguous_cpu_view():
    """
    Feature: CPU real and imag views.
    Description: Read both components from a transposed complex Tensor in PyNative mode.
    Expectation: Values, strides, and data addresses retain the input View layout.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    data = np.array([[1 + 2j, 3 + 4j, 5 + 6j],
                     [7 + 8j, 9 + 10j, 11 + 12j]], dtype=np.complex64)
    x = mint.permute(Tensor(data), (1, 0))
    real = mint.real(x)
    imag = mint.imag(x)

    np.testing.assert_array_equal(real.asnumpy(), data.T.real)
    np.testing.assert_array_equal(imag.asnumpy(), data.T.imag)
    assert tuple(x.stride()) == (1, 3)
    assert tuple(real.stride()) == tuple(imag.stride()) == (2, 6)
    assert real.data_ptr() == x.data_ptr()
    assert imag.data_ptr() - real.data_ptr() == np.dtype(np.float32).itemsize


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_real_non_complex_cpu_view():
    """
    Feature: CPU real view of a non-complex Tensor.
    Description: Call mint.real on a float Tensor in PyNative mode.
    Expectation: The result keeps the input values and shares its data address.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    data = np.array([1.25, -2.5, 3.0], dtype=np.float32)
    x = Tensor(data)
    real = mint.real(x)

    np.testing.assert_array_equal(real.asnumpy(), data)
    assert real.dtype == x.dtype
    assert real.data_ptr() == x.data_ptr()
