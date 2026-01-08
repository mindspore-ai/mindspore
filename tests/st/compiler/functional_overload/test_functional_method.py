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
"""Test the feature of functional"""
# pylint: disable=redefined-builtin
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_method_clamp():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min, max):
        return x.clamp(min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out_clamp_tensor = func(x, ms.Tensor(2), ms.Tensor(4))
    out_clamp_scalar = func(x, 2, 4)
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out_clamp_tensor.asnumpy() == expect.asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_default():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min):
        return x.clamp(min)

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    expect = ms.Tensor([2, 2, 3, 4, 5])
    assert np.all(out.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_any():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min, max):
        return x.clamp(min, ms.Tensor(max.asnumpy()))

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), ms.Tensor(4))
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min):
        return x.clamp(min, max=ms.Tensor(4)), x.clamp(max=3, min=1)

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([1, 2, 3, 3, 3]).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword_any():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min, max):
        return x.clamp(min=ms.Tensor(min.asnumpy())), x.clamp(2, max=int(max.asnumpy()))

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), ms.Tensor(4))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 5]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.max()

    x = ms.Tensor([1, 2, 3, 4, 5])
    assert func(x).asnumpy() == ms.Tensor(5).asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.max(keepdims=False)

    x = ms.Tensor([1, 2, 3, 4, 5])
    assert func(x).asnumpy() == ms.Tensor(5).asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max_keyword_any():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @ms.jit
    def func(x, y):
        return x.max(return_indices=bool(y.asnumpy()))

    x = ms.Tensor([1, 2, 3, 4, 5])
    y = ms.Tensor(False)
    assert func(x, y).asnumpy() == ms.Tensor(5).asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_min_exception():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.min(None, False, None)

    with pytest.raises(TypeError):
        func(ms.Tensor([1, 2, 3, 4, 5]))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_transpose():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.transpose.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.transpose((1, 0))

    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    out = func(x)
    expect = ms.Tensor(np.array([[1, 4], [2, 5], [3, 6]]))
    assert np.all(out.asnumpy() == expect.asnumpy())

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_add_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, y):
        return Tensor.add(x, y)

    with pytest.raises(TypeError):
        func([1], [2])

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_add_type_error_002():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return Tensor.add(x)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(np.array([2]).astype(np.float32)))

    assert "The valid calling should be:" in str(raise_info.value)
    assert "Tensor.add(other=<number>, *, alpha=<number>)" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_add_type_error_003():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, y):
        return x.add(y, 1)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(np.array([2]).astype(np.float32)), Tensor(np.array([2]).astype(np.float32)))

    assert "match failed because invalid types: (Tensor, int)" in str(raise_info.value)
    assert "~~~~~~~~~~~~~" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_add_type_error_004():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, y):
        return x.add(y)

    with pytest.raises(TypeError):
        func(Tensor(2), '1')

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_all_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return Tensor.all(x, keep_dims=True)

    with pytest.raises(TypeError):
        func(1)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_clamp_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.clamp(min=1, max_value=10)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), ms.float32))

    assert "match failed because incorrect keyword name: max_value" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_div_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, y):
        return x.div(y, rounding_mode='floor', mode='floor')

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), Tensor(np.array([4.0, 5.0, 6.0]), ms.float32))

    assert "The valid calling should be:" in str(raise_info.value)
    assert "Tensor.div(other=<number,Tensor>)" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_log_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.log(1)

    with pytest.raises(TypeError):
        func(Tensor(np.array([1.0, 2.0, 4.0]), ms.float32))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_mean_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, y):
        return Tensor.mean(x, y, keepdim=0)

    with pytest.raises(TypeError):
        func('1', '0')

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_mean_type_error_002():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, y):
        return Tensor.mean(x, y, 0)

    with pytest.raises(TypeError):
        func('1', '0')

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_fault_tensor_roll_type_error_001():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.roll(shifts=2, dims=1.1)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32)))

    assert "match failed because invalid types: (shifts=int, dims=float)" in str(raise_info.value)
    assert "~~~~~~~~~~~~~~~~~~~~~~~~" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_too_many_parameters():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.clamp(1, [1], 1)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "Tensor.clamp(min=<number,None>, max=<number,None>)" in str(raise_info.value)
    assert "Tensor.clamp(min=<Tensor,None>, max=<Tensor,None>)" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_keyword_name():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.max(a=1, b=2)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "match failed because incorrect keyword name: a" in str(raise_info.value)
    assert "Tensor.max()" in str(raise_info.value)
    assert "Tensor.max(dim=<int>, keepdim=<bool>)" in str(raise_info.value)
    assert ("Tensor.max(axis=<int,Tuple,None,List>, keepdims=<bool>, *, initial=<number,None>, where=<bool,Tensor>, "
            "return_indices=<bool>)") in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_type():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.max(1, [1])

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "    match failed because invalid types: (int, list<int>)" in str(raise_info.value)
    assert "                                              ~~~~~~~~~~~" in str(raise_info.value)
    assert "Tensor.max()" in str(raise_info.value)
    assert "Tensor.max(dim=<int>, keepdim=<bool>)" in str(raise_info.value)
    assert ("Tensor.max(axis=<int,Tuple,None,List>, keepdims=<bool>, *, initial=<number,None>, where=<bool,Tensor>, "
            "return_indices=<bool>)") in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_args_num_of_1_candidate():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.abs(1)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "abs() takes 0 positional arguments but 1 was given." in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_args_num_of_1_candidate_2():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.less()

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "missing 1 required" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_type_of_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.max(0, True, initial=[1])

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "max(): argument 'initial' must be Number but got list of Any" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_type_of_1_candidate():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.max(0, 1, initial=None)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "max(): argument 'keepdims' (position 2) must be bool, not int." in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_wrong_keyword_name_of_1_candidate():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.max(1, True, il=2)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "match failed because incorrect keyword name: il" in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_args_to_string():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.max([1, "2", (1, 2, 3), [1, 4, "3"]], (1, "2", (1, 2, 3), [1, 4, "3"]))

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert ("list<int, string, tuple<int, int, int>, list<int, int, string>>, tuple<int, string, tuple<int, int, int>, "
            "list<int, int, string>>") in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_reshape():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.reshape.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.reshape(1, 2, 3)

    x = ms.Tensor([[[1], [2]], [[3], [4]], [[5], [6]]])
    out = func(x)
    expect = ms.Tensor([[[1, 2, 3], [4, 5, 6]]])
    assert np.all(out.asnumpy() == expect.asnumpy())

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_implicit_tuple():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.reshape(1, 1.1)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "reshape(): argument 'shape' (position 2) must be tuple of int, not float." in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_method_implicit_tuple_2():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.reshape(1)

    out = func(Tensor(1))
    expected = Tensor(1)

    assert np.all(out.asnumpy() == expected.asnumpy())

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_method_implicit_tuple_3():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.reshape(1.1)

    with pytest.raises(TypeError) as raise_info:
        func(Tensor(1))

    assert "reshape(): argument 'shape' (position 1) must be tuple of int, not float." in str(raise_info.value)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_method_explicit_tuple():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x):
        return x.reshape((1, 2, 3))

    x = ms.Tensor([[[1], [2]], [[3], [4]], [[5], [6]]])
    out = func(x)
    expected = ms.Tensor([[[1, 2, 3], [4, 5, 6]]])
    assert np.all(out.asnumpy() == expected.asnumpy())
