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
"""test case of tensor set_ function"""

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import nn
from mindspore import Tensor, mint
from mindspore.graph.api import _pynative_executor

class Set(nn.Cell):
    def construct(self, dst):
        dst.set_()
        return dst

class SetWithTensorOrStorage(nn.Cell):
    def construct(self, dst, src):
        dst.set_(src)
        return dst

class SetWithTensorOrStorageCustom(nn.Cell):
    def construct(self, dst, src, storage_offset, shape, stride=None):
        if stride is None:
            dst.set_(src, storage_offset, shape)
        else:
            dst.set_(src, storage_offset, shape, stride)
        return dst


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_set_(mode):
    """
    Feature: Tensor.set_
    Description: Verify the result of Tensor.set_
    Expectation: success
    """
    ms.set_context(mode=mode)

    set_func_with_no_args = Set()
    set_func_storage = SetWithTensorOrStorage()
    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([], dtype=ms.int32)
    np_input = np.arange(3*5*7*9).reshape(3, 5, 7, 9)
    t2 = Tensor(np_input, dtype=ms.int32)
    t3 = Tensor(np_input, dtype=ms.int32)

    # test set_()
    t3 = set_func_with_no_args(t3)
    assert np.allclose(t3.numel(), 0)

    # test set_(tensor source)
    t1 = set_func_storage(t1, t2)
    assert np.allclose(t1.data_ptr(), t2.data_ptr())

    # test set(storage source, int storage_offset, tuple shape, tuple stride=None)
    shape = (9, 5, 7, 3)
    t1 = set_func_storage_custom(t1, t2.storage(), 0, shape)
    assert np.allclose(t1.shape, shape)

    # test set(storage source, int storage_offset, list shape, tuple stride=None)
    shape = [9, 5, 7, 3]
    t1 = set_func_storage_custom(t1, t2.storage(), 0, shape)
    assert np.allclose(t1.shape, shape)

    # test stride after set_
    stride = [105, 21, 3, 1]
    assert np.allclose(t1.stride(), stride)

    # test not contiguous
    stride = [110, 25, 5, 1]
    t1 = set_func_storage_custom(t1, t2.storage(), 0, shape, stride)
    assert np.allclose(t1.stride(), stride)

    # test argument names
    stride = [105, 21, 3, 1]
    t1.set_(t2.storage(), 0, size=shape, stride=stride)
    assert np.allclose(t1.shape, shape)
    assert np.allclose(t1.stride(), stride)

    # when source is tensor
    t1 = Tensor([], dtype=ms.int32)
    t1.set_(source=t2)
    assert np.allclose(t1.data_ptr(), t2.data_ptr())

    # when source is storage
    t1.set_(source=t2.storage())
    assert np.allclose(t1.data_ptr(), t2.data_ptr())

    # when source is storage and other args also specified
    t1.set_(source=t2.storage(), storage_offset=0, size=shape, stride=stride)
    assert np.allclose(t1.shape, shape)
    assert np.allclose(t1.stride(), stride)

    # if shape changes, set_ will resize the storage inplace
    storage_offset = 0
    shape = (2, 2)
    t1 = Tensor([1, 2, 3, 4], dtype=ms.int32)
    t1 = set_func_storage_custom(t1, t1.storage(), storage_offset, shape)
    assert np.allclose(t1.storage_offset(), storage_offset)
    assert np.allclose(t1.storage().nbytes(), (storage_offset + shape[0] * shape[1]) * 4)

    #test bool value
    t1 = Tensor([True, True], dtype=ms.bool)
    t2 = Tensor([False, False], dtype=ms.bool)
    t1 = set_func_storage(t1, t2)
    assert np.allclose(t1.data_ptr(), t2.data_ptr())


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_set_errors(mode):
    """
    Feature: Tensor.set_
    Description: Verify the result of Tensor.set_ exception
    Expectation: success
    """
    ms.set_context(mode=mode)

    set_func_storage = SetWithTensorOrStorage()
    set_func_storage_custom = SetWithTensorOrStorageCustom()

    np_input = np.arange(3*5*7*9).reshape(3, 5, 7, 9)
    t1 = Tensor(np_input, dtype=ms.int32)
    t2 = Tensor([1, 2, 3], dtype=ms.int32)
    t3 = Tensor([1, 2, 3], dtype=ms.int32)

    # when offer shape ande stride are same as dst shape and stride
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t1, t2, 0, t1.shape)
    assert "are out of bounds for storage" in str(err.value)

    # provide invalid storage offset
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t2, t1, -1, t1.shape)
    assert "invalid storage offset" in str(err.value)

    # provide invalid shape
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t2, t1, 0, (2, -1, 2))
    assert "Storage size calculation overflowed" in str(err.value)

    # provide invalid stride
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t2, t1, 0, (2, 2), (-2, 1))
    assert "Storage size calculation overflowed" in str(err.value)

    # provide shape.size != stride.size
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t2, t1, 0, (2, 2), (2, 2, 1))
    assert "unequal shape length" in str(err.value)

    # provide shape.size > 8
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t2, t1, 0, (2, 2, 2, 2, 2, 2, 2, 2, 2))
    assert "The input shape's dim must in the range of [0, 8]" in str(err.value)

    # test invalid args : cannot fix any
    with pytest.raises(TypeError) as err:
        _ = t2.set_(t1, 0)
    assert "Failed calling set_ with" in str(err.value)
    # test invalid args : can fix one but not all args
    with pytest.raises(TypeError) as err:
        _ = t2.set_(t1, 0, 0)
    assert "Failed calling set_ with" in str(err.value)

    # in signature Set(tensor source, int storage_offset, tuple[int]|list[int] shape, tuple[int]|list[int] stride=None)
    t2 = set_func_storage_custom(t2, t1, 0, (2, 2), (3, 1))
    with pytest.raises(RuntimeError) as err:
        _ = set_func_storage_custom(t3, t2, 0, (2, 2), (2, 1))
    assert "passed in tensor to be used as storage must be contiguous" in str(err.value)

    # change dtype
    np_input = np.arange(2*3).reshape(2*3)
    f_cpu = Tensor(np_input, dtype=ms.float32, device='CPU')
    d_cpu = Tensor(np_input, dtype=ms.float64, device='CPU')

    with pytest.raises(RuntimeError):
        _ = set_func_storage(f_cpu, d_cpu.storage())

    with pytest.raises(RuntimeError):
        _ = set_func_storage(f_cpu, d_cpu)

    with pytest.raises(RuntimeError):
        _ = set_func_storage_custom(f_cpu, d_cpu.storage(), 0, d_cpu.shape, d_cpu.stride())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_set_errors_with_different_device(mode):
    """
    Feature: Tensor.set_
    Description: Verify the result of Tensor.set_ exception of diff device
    Expectation: success
    """
    ms.set_context(mode=mode)

    set_func_storage = SetWithTensorOrStorage()
    set_func_storage_custom = SetWithTensorOrStorageCustom()

     # test change device error
    np_input = np.arange(2*3).reshape(2*3)
    f_cpu = Tensor(np_input, dtype=ms.float32, device='CPU')
    f_ascend = Tensor(np_input, dtype=ms.float32, device='CPU')
    f_ascend = f_ascend.to('Ascend')

    # cpu -> ascend
    with pytest.raises(RuntimeError):
        _ = set_func_storage(f_cpu, f_ascend.storage())

    with pytest.raises(RuntimeError):
        _ = set_func_storage(f_cpu, f_ascend)

    with pytest.raises(RuntimeError):
        _ = set_func_storage_custom(f_cpu, f_ascend.storage(), 0, f_ascend.shape, f_ascend.stride())

    # ascend -> cpu
    with pytest.raises(RuntimeError):
        _ = set_func_storage(f_ascend, f_cpu.storage())

    with pytest.raises(RuntimeError):
        _ = set_func_storage(f_ascend, f_cpu)

    with pytest.raises(RuntimeError):
        _ = set_func_storage_custom(f_ascend, f_cpu.storage(), 0, f_cpu.shape, f_cpu.stride())


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_copy_after_set_(mode):
    """
    Feature: Tensor.set_
    Description: Verify the inplace copy result after Tensor.set_
    Expectation: success
    """

    ms.set_context(mode=mode)

    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([1, 2, 3, 4, 5], dtype=ms.int32)
    storage = t1.storage()
    shape = t1.shape
    stride = t1.stride()
    t2 = mint.empty((0,), dtype=t1.dtype, device=t1.device)

    t2 = set_func_storage_custom(t2, storage, 0, shape, stride)
    assert np.allclose(t1.asnumpy(), t2.asnumpy())

    t3 = Tensor([6, 7, 8, 9, 10], dtype=ms.int32)

    # test inplace copy_ after set_
    t2.copy_(t3)
    assert np.allclose(t2.asnumpy(), t3.asnumpy())
    assert np.allclose(t1.asnumpy(), t3.asnumpy())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_copy_errors_after_set_with_unequal_mem_size(mode):
    """
    Feature: Tensor.set_
    Description: Verify the CPU inplace copy exception result after Tensor.set_ with unequal_mem_size of copy side
    Expectation: success
    """
    ms.set_device('CPU')
    ms.set_context(mode=mode)

    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=ms.int32)
    storage = t1.storage()
    shape = (6,)
    stride = t1.stride()
    t2 = mint.empty((0,), dtype=t1.dtype, device=t1.device)

    t2 = set_func_storage_custom(t2, storage, 0, shape, stride)

    t3 = Tensor([6, 7, 8, 9, 10, 11], dtype=ms.int32)

    # test inplace copy_ exception after set_ with unequal_mem_size of copy side
    with pytest.raises(RuntimeError) as err:
        t2.copy_(t3)
        _pynative_executor.sync()
    assert "an unexpected kernel has been launched" in str(err.value)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_setitem_after_set_cpu(mode):
    """
    Feature: Tensor.set_
    Description: Verify setitem result after Tensor.set_ on CPU
    Expectation: success
    """
    ms.set_device('CPU')
    ms.set_context(mode=mode)

    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([1, 2, 3, 4, 5, 6], dtype=ms.int32)
    storage = t1.storage()
    shape = (2, 2)
    stride = (2, 1)
    t2 = mint.empty((0,), dtype=t1.dtype, device=t1.device)

    t2 = set_func_storage_custom(t2, storage, 0, shape, stride)
    t2[0:2, 0:2] = 10

    # test setitem
    np_t2 = np.array([[10, 10], [10, 10]])
    assert np.allclose(t2.asnumpy(), np_t2)

    # test modify value of t2 also changes value of t1
    np_t1 = np.array([10, 10, 10, 10, 5, 6])
    assert np.allclose(t1.asnumpy(), np_t1)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_setitem_after_set_ascend(mode):
    """
    Feature: Tensor.set_
    Description: Verify setitem result after Tensor.set_ on Ascend
    Expectation: success
    """
    ms.set_context(mode=mode)

    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([1, 2, 3, 4, 5, 6], dtype=ms.int32)
    t1 = t1.to('Ascend')
    storage = t1.storage()
    shape = (2, 2)
    stride = (2, 1)
    t2 = mint.empty((0,), dtype=t1.dtype, device='Ascend')

    t2 = set_func_storage_custom(t2, storage, 0, shape, stride)
    t2[0:2, 0:2] = 10

    # test setitem
    np_t2 = np.array([[10, 10], [10, 10]])
    assert np.allclose(t2.asnumpy(), np_t2)

    # test modify value of t2 also changes value of t1
    np_t1 = np.array([10, 10, 10, 10, 5, 6])
    assert np.allclose(t1.asnumpy(), np_t1)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_add_after_set_with_lazy_copy(mode):
    """
    Feature: Tensor.set_
    Description: Verify the Ascend inplace add result after Tensor.set_
    Expectation: success
    """

    ms.set_device('Ascend')
    ms.set_context(mode=mode)

    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([1, 2, 3, 4, 5], dtype=ms.int32)
    storage = t1.storage()
    shape = t1.shape
    stride = t1.stride()
    t2 = Tensor([1], dtype=ms.int32)

    t2 = set_func_storage_custom(t2, storage, 0, shape, stride)
    assert np.allclose(t1.asnumpy(), t2.asnumpy())

    np_add = np.array([2, 3, 4, 5, 6])

    # test inplace add_ after set_
    t2.add_(1)

    assert np.allclose(t2.asnumpy(), np_add)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_add_after_set_with_lazy_copy_error(mode):
    """
    Feature: Tensor.set_
    Description: Verify the Ascend inplace add error after Tensor.set_
    Expectation: success
    """

    ms.set_device('Ascend')
    ms.set_context(mode=mode)

    set_func_storage_custom = SetWithTensorOrStorageCustom()

    t1 = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=ms.int32)
    storage = t1.storage()
    shape = (5,)
    stride = t1.stride()
    t2 = Tensor([1], dtype=ms.int32)

    t2 = set_func_storage_custom(t2, storage, 0, shape, stride)
    np_set = np.array([1, 2, 3, 4, 5])
    assert np.allclose(t2.asnumpy(), np_set)

    # test inplace add_ error after set_
    with pytest.raises(RuntimeError) as err:
        t2.add_(1)
        _pynative_executor.sync()
    assert "Invalid sync host to device" in str(err.value)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_set_increase_storage_size_ascend(mode):
    """
    Feature: Tensor.set_
    Description: Verify the custom size larger than origin size and scale up
    Expectation: success
    """
    ms.set_context(mode=mode)

    t1 = Tensor([1, 2, 3, 4, 5], dtype=ms.float32).to('Ascend')
    t2 = Tensor([0], dtype=ms.float32).to('Ascend')
    t3 = Tensor([3], dtype=ms.float32).to('Ascend')

    t2.set_(t1.storage())
    t3.set_(t2.storage(), 0, (10,))

    # test scale up
    expected_size = 40
    assert np.allclose(t3.storage().nbytes(), expected_size)

    # test shared memory
    t3[4] = 99.
    assert np.allclose(t1[4].asnumpy(), t3[4].asnumpy())
    assert np.allclose(t2[4].asnumpy(), t3[4].asnumpy())

    assert np.allclose(t1.data_ptr(), t2.data_ptr())
    assert np.allclose(t1.data_ptr(), t3.data_ptr())
    assert np.allclose(t2.data_ptr(), t3.data_ptr())


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_set_increase_storage_size_cpu(mode):
    """
    Feature: Tensor.set_
    Description: Verify the custom size larger than origin size and scale up
    Expectation: success
    """
    ms.set_context(mode=mode)

    t1 = Tensor([1, 2, 3, 4, 5], dtype=ms.float32)
    t2 = Tensor([0], dtype=ms.float32)
    t3 = Tensor([3], dtype=ms.float32)

    t2.set_(t1)
    t3.set_(t2, 0, (10,))

    # test shared memory
    t3[4] = 99.
    assert np.allclose(t1[4].asnumpy(), t3[4].asnumpy())
    assert np.allclose(t2[4].asnumpy(), t3[4].asnumpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_set_expand_size_not_memory_leak(mode):
    """
    Feature: Tensor.set_
    Description: Verify that expanding the tensor size does not cause memory leak
    Expectation: success
    """
    def scope():
        origin_tensor = Tensor(np.random.rand(1, 3, 224, 224), dtype=ms.float32).to('Ascend')
        expand_size = 1 * 3 * 224 * 224 + 1
        expand_tensor = Tensor([0], dtype=ms.float32).to('Ascend')
        expand_tensor.set_(origin_tensor, 0, (expand_size,))

    ms.set_context(mode=mode)
    for _ in range(10):
        ms.runtime.synchronize()
        alloc_mem = ms.runtime.memory_allocated()
        assert np.allclose(0, alloc_mem)
        scope()
        after_set_mem = ms.runtime.memory_allocated()
        assert np.allclose(0, after_set_mem)
