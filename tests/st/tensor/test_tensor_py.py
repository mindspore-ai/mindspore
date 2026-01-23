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
import os
import numpy as np
import pytest
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import Tensor
from mindspore.common.initializer import One
from mindspore._c_expression import ParamInfo
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_shape(mode):
    """
    Feature: get or set tensor's shape.
    Description: test tensor.shape function.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    shape = x.shape
    assert shape == (1, 2, 3)

    # error info: List items must be integers
    with pytest.raises(TypeError):
        x.shape = ['a', 3]
        _pynative_executor.sync()

    # error info: Expected a Python list
    with pytest.raises(TypeError):
        x.shape = (1, 3)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_const_arg(mode):
    """
    Feature: get or set tensor's const_arg_flag_.
    Description: test tensor.set_const_arg function.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    # error info: The init_flag property value must be a boolean
    with pytest.raises(TypeError):
        x.const_arg = 123
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_is_pinned(mode):
    """
    Feature: check whether tensor is on pinned memory.
    Description: test tensor.is_pinned function.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(shape=(1, 2, 3), dtype=mstype.float32, init=One())
    assert not x.is_pinned()

    y = Tensor([[1, 2, 3]])
    assert not y.is_pinned()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_init_flag(mode):
    """
    Feature: check whether tensor's init_flag is True.
    Description: test tensor.init_flag.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    assert not x.init_flag
    assert not x.is_init()

    x.init_flag = True
    assert x.init_flag
    assert x.is_init()

    x.set_init_flag(False)
    assert not x.init_flag
    assert not x.is_init()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_param_info(mode):
    """
    Feature: check tensor's param info.
    Description: test tensor.param_info.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    print(x.param_info)

    pi = ParamInfo()
    pi.name = "new_name"
    x.param_info = pi
    assert x.param_info.name == "new_name"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_dtype(mode):
    """
    Feature: check tensor's dtype.
    Description: test tensor.dtype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    assert x.dtype == mstype.float32

    x._dtype = mstype.float16 # pylint:disable=protected-access
    assert x.dtype == mstype.float16


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_offload(mode):
    """
    Feature: check tensor offload.
    Description: test tensor.offload.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    filepath = './test_data.txt'
    res = x.offload(filepath)
    is_file_exist = os.path.isfile(filepath)
    assert res
    assert is_file_exist


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_is_signed(mode):
    """
    Feature: check whether tensor is signed.
    Description: test tensor.is_signed().
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.int32)
    assert x.is_signed()

    y = Tensor(shape=(1, 3), dtype=mstype.uint32, init=One())
    assert not y.is_signed()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_tostring(mode):
    """
    Feature: get tensor info in string format.
    Description: test tensor.__str__().
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=mstype.int32)
    x_str = str(x)
    assert '[[1 2 3]]' in x_str

    x_repr = x.__repr__()
    assert "Tensor(shape=[1, 3], dtype=Int32, value=\n[[1, 2, 3]])" in x_repr

    y = Tensor(shape=(1, 3), dtype=mstype.int32, init=One())
    y_str = str(y)
    assert 'Tensor(shape=[1, 3], dtype=Int32, value=\n<uninitialized>)' in y_str


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_requires_grad(mode):
    """
    Feature: check whether tensor needs grad.
    Description: test tensor._requires_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=mstype.int32)
    assert not x._requires_grad # pylint:disable=protected-access

    x._requires_grad = True # pylint:disable=protected-access
    assert x._requires_grad # pylint:disable=protected-access

    # error info: The requires_grad property value must be a boolean
    with pytest.raises(TypeError):
        x._requires_grad = 123 # pylint:disable=protected-access
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_data_sync(mode):
    """
    Feature: check tensor's data sync.
    Description: test tensor.data_sync.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=mstype.int32)
    assert x.data_sync() is None


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.int8, ms.uint8, ms.int16, ms.uint16, ms.uint16, ms.int32, ms.uint32,
                                   ms.int64, ms.uint64])
def test_tensor_tolist1(mode, dtype):
    """
    Feature: convert tensor to list.
    Description: test tensor._tolist.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=dtype)
    output = x.tolist()
    expect_output = [[1, 2, 3]]
    assert np.allclose(output, expect_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.float64, ms.bfloat16])
def test_tensor_tolist2(mode, dtype):
    """
    Feature: convert tensor to list.
    Description: test tensor._tolist.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    output = x.tolist()
    expect_output = [[1.0, 2.0, 3.0]]
    assert np.allclose(output, expect_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.bool])
def test_tensor_tolist3(mode, dtype):
    """
    Feature: convert tensor to list.
    Description: test tensor._tolist.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[True, False]], dtype=dtype)
    output = x.tolist()
    expect_output = [[True, False]]
    assert output == expect_output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.complex128])
def test_tensor_tolist4(mode, dtype):
    """
    Feature: convert tensor to list.
    Description: test tensor._tolist.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=dtype)
    output = x.tolist()
    expect_output = [[1+0j, 2+0j, 3+0j]]
    assert np.allclose(output, expect_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_operator_assign_value(mode):
    """
    Feature: assign value for tensor.
    Description: test tensor assign value.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=ms.int32)
    y = x
    assert np.allclose(y, x)
    assert y.dtype == x.dtype
    assert y.nbytes == x.nbytes


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_storage(mode):
    """
    Feature: resize tensor's storage.
    Description: test tensor.storage().resize_().
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1, 2, 3]], dtype=mstype.int32)
    # The implementation of storage().element_size() always return 1. It should return the number of bytes of dtype,
    # it be corrected in future.
    assert x.storage().element_size() == 1
    assert x.storage().__len__() == 12
    assert x.storage().nbytes() == 12

    x.storage().resize_(0)
    assert x.storage().__len__() == 0
    assert x.storage().nbytes() == 0

    x.storage().resize_(12)
    assert x.storage().__len__() == 12
    assert x.storage().nbytes() == 12

    # error info: The function __getitem__ is not implemented for Storage
    with pytest.raises(RuntimeError):
        x_storage_slice = x.storage()[0]
        expect_output = 1
        assert np.allclose(x_storage_slice, expect_output)

    # error info: The function __setitem__ is not implemented for Storage
    with pytest.raises(RuntimeError):
        x.storage()[0] = 10


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_storage_gpu(mode):
    """
    Feature: resize tensor's storage.
    Description: test tensor.storage().resize_().
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, device_target='GPU')
    x = Tensor([[1, 2, 3]], dtype=mstype.int32)
    print(f'x.device:{x.device}')

    x = x + 1
    print(f'after add.x.device:{x.device}')
    print(f'test_tensor_storage_gpu.nbytes:{x.storage().nbytes()}')

    # error info: Current Storage only support NPU, but got GPU
    with pytest.raises(RuntimeError):
        x.storage().resize_(0)
