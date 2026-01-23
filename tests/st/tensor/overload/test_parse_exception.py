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
"""Test the overload functional method exception info"""
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import numpy as np
ms.set_context(mode=ms.PYNATIVE_MODE)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_extra_args_exception():
    """
    Feature: parse error info
    Description: Check parse error info when got extra argument.
    Expectation: Raise expected exception.
    """
    input_x = ms.Tensor([1, 2, 3, 4, 5])
    with pytest.raises(TypeError) as raise_info:
        input_x.abs(1)
    assert "abs()" in str(raise_info.value)
    assert "takes 0 positional arguments but 1 was given" in str(raise_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_missing_args_exception():
    """
    Feature: parse error info
    Description: Check parse error info when missing argument.
    Expectation: Raise expected exception.
    """
    input_x = ms.Tensor([1, 2, 3, 4, 5])
    with pytest.raises(TypeError) as raise_info:
        input_x.equal()
    assert "equal()" in str(raise_info.value)
    assert "missing 1 required positional argument: 'other'" in str(raise_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_type_error_exception():
    """
    Feature: parse type error
    Description: Check parse error info when got wrong type.
    Expectation: Raise expected exception.
    """
    input_x = ms.Tensor([1, 2, 3, 4, 5])
    with pytest.raises(TypeError) as raise_info:
        input_x.max(1, True, initial=[1])
    assert "max()" in str(raise_info.value)
    assert "argument 'initial' must be Number but got list" in str(raise_info.value)

    with pytest.raises(TypeError) as raise_info:
        input_x.reshape(1.1)
    assert "reshape()" in str(raise_info.value)
    assert "argument 'shape' (position 1) must be tuple of int \
but found type of float at pos 0" in str(raise_info.value)

    with pytest.raises(TypeError) as raise_info:
        input_x.max(1, 1, initial=None)
    assert "max()" in str(raise_info.value)
    assert "argument 'keepdims' (position 2) must be bool, not int" in str(raise_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_kw_args_exception():
    """
    Feature: parse error info of wrong kw_args.
    Description: Check parse error info when got wrong kw_args.
    Expectation: Raise expected exception.
    """
    input_x = ms.Tensor([1, 2, 3, 4, 5])
    with pytest.raises(TypeError) as raise_info:
        input_x.max(1, True, il=[1])
    assert "max()" in str(raise_info.value)
    assert "got an unexpected keyword argument 'il'" in str(raise_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_multiple_sig_exception():
    """
    Feature: parse error info.
    Description: Check parse error info when multiple overloads.
    Expectation: Raise expected exception.
    """
    input_x = ms.Tensor([1, 2, 3, 4, 5])
    with pytest.raises(TypeError) as raise_info:
        input_x.max(a=1, b=1)
    assert "Failed calling max with" in str(raise_info.value)
    assert "max(dim=<int>, keepdim=<bool>)" in str(raise_info.value)
    assert "incorrect keyword name: a, b" in str(raise_info.value)
    assert ("max(axis=<None, int, list of int, tuple of int>, keepdims=<bool>, *, "
            "initial=<None, Number>, where=<Tensor, bool>, return_indices=<bool>)") in str(raise_info.value)
    assert "incorrect keyword name: a, b" in str(raise_info.value)

    with pytest.raises(TypeError) as raise_info:
        input_x.max(1, [1])
    assert "Failed calling max with" in str(raise_info.value)
    assert "max(dim=<int>, keepdim=<bool>)" in str(raise_info.value)
    assert "(int, List<int>)" in str(raise_info.value)
    assert "~~~~~~~~~~~" in str(raise_info.value)

    x = ms.Tensor(np.random.rand(2, 2))
    with pytest.raises(TypeError) as raise_info:
        x.clamp([1], 1)
    assert "Tensor.clamp(min=<None, Tensor>, max=<None, Tensor>)" in str(raise_info.value)
    assert "Tensor.clamp(min=<None, Number>, max=<None, Number>)" in str(raise_info.value)
