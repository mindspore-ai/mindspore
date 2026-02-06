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
"""Test the feature of functional"""
# pylint: disable=redefined-builtin
import pytest
import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_executed_by_graph_mode, pi_jit_with_config


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=3072.01)
def test_method_clamp():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min, max):
        return x.clamp(min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out_clamp_tensor = func(x, ms.Tensor(2), ms.Tensor(4))
    out_clamp_scalar = func(x, 2, 4)
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out_clamp_tensor.asnumpy() == expect.asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_default():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min):
        return x.clamp(min)

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    expect = ms.Tensor([2, 2, 3, 4, 5])
    assert np.all(out.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min):
        return x.clamp(min, max=ms.Tensor(4)), x.clamp(max=3, min=1)

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([1, 2, 3, 3, 3]).asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x):
        return x.max()

    x = ms.Tensor([1, 2, 3, 4, 5])
    assert func(x).asnumpy() == ms.Tensor(5).asnumpy()
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x):
        return x.max(keepdims=False)

    x = ms.Tensor([1, 2, 3, 4, 5])
    assert func(x).asnumpy() == ms.Tensor(5).asnumpy()
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_min_exception():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Raise expected exception.
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x):
        return x.min(None, False, None)

    with pytest.raises(TypeError):
        func(ms.Tensor([1, 2, 3, 4, 5]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_reshape():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.reshape.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x):
        return x.reshape(1, 2, 3)

    x = ms.Tensor([[[1], [2]], [[3], [4]], [[5], [6]]])
    out = func(x)
    expect = ms.Tensor([[[1, 2, 3], [4, 5, 6]]])
    assert np.all(out.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_transpose():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.transpose.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x):
        return x.transpose(1, 0)

    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    out = func(x)
    expect = ms.Tensor(np.array([[1, 4], [2, 5], [3, 6]]))
    assert np.all(out.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)
