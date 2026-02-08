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
import mindspore as ms
from mindspore import Tensor, default_generator
import pytest
import numpy as np
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_get_storage():
    """
    Feature: get_storage
    Description: Verify the result of get_storage
    Expectation: success
    """
    a = Tensor(1.0)
    b = a * 1
    storage = b.untyped_storage()

    assert storage.size() == 4
    assert storage.nbytes() == 4


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_resize():
    """
    Feature: storage resize
    Description: Verify the result after resize storage
    Expectation: success
    """
    a = Tensor(1.0)
    b = a * 1
    storage = b.untyped_storage()

    assert storage.size() == 4
    assert storage.nbytes() == 4

    storage.resize_(0)

    assert storage.size() == 0
    assert storage.nbytes() == 0
    assert storage.data_ptr() == 0

    storage.resize_(4)
    assert storage.size() == 4
    assert storage.nbytes() == 4
    assert storage.data_ptr() != 0

    storage.resize_(12)
    assert storage.size() == 12
    assert storage.nbytes() == 12
    assert storage.data_ptr() != 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_copy():
    """
    Feature: storage copy
    Description: Verify the result after copy
    Expectation: success
    """
    a = Tensor(2.0)
    b = a * 1
    c = a * 3
    storage_b = b.untyped_storage()
    storage_c = c.untyped_storage()

    assert b.item() == 2.0

    storage_b.copy_(storage_c)

    assert b.item() == 6.0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_init_tensor():
    """
    Feature: init
    Description: Verify the result of untyped_storage
    Expectation: success
    """
    a = Tensor(2.0)
    assert a.untyped_storage().size() == 4


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_cpu_and_npu_copy():
    """
    Feature: copy between cpu and npu
    Description: Verify the result of get_storage
    Expectation: success
    """
    a = Tensor(2.0)
    b = Tensor(2.0) * 2.0
    a.untyped_storage().copy_(b.untyped_storage())
    assert a.item() == 4.0

    a = Tensor(2.0)
    b = Tensor(2.0) * 2.0
    b.untyped_storage().copy_(a.untyped_storage())
    assert b.item() == 2.0

    a = Tensor(2.0)
    b = Tensor(2.0) * 2.0
    b.untyped_storage().copy_(a.untyped_storage(), non_blocking=True)
    ms.runtime.synchronize()
    assert b.item() == 2.0

    a = Tensor(np.ones(1000))
    b = Tensor(np.ones(1000)) * 3
    b.untyped_storage().copy_(a.untyped_storage())
    assert b.sum().item() == 1000


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_view():
    """
    Feature: view of storage
    Description: Verify view tensor value
    Expectation: success
    """
    a = Tensor([2.0, 2.0])
    b = a[0]
    c = Tensor([3.0, 3.0])
    a.untyped_storage().copy_(c.untyped_storage())

    assert b.item() == 3.0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_copy_exception():
    """
    Feature: Exception when two storage size not equal
    Description: Check exception
    Expectation: success
    """
    a = Tensor([2.0, 2.0, 2.0])
    b = Tensor([3.0, 3.0])
    with pytest.raises(RuntimeError):
        a.untyped_storage().copy_(b.untyped_storage())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_cpu_after_compute():
    """
    Feature: cpu storage
    Description: Verify the result of storage
    Expectation: success
    """
    ms.set_device("CPU")
    a = Tensor(2.0)
    a = a * 1

    assert a.untyped_storage().size() == 4


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_resize_cpu():
    """
    Feature: resize storage
    Description: Verify the result after resize storage
    Expectation: success
    """
    ms.set_device("CPU")
    a = Tensor(1.0)
    b = a * 1
    storage = b.untyped_storage()

    assert storage.size() == 4
    assert storage.nbytes() == 4

    storage.resize_(0)

    assert storage.size() == 0
    assert storage.nbytes() == 0
    assert storage.data_ptr() == 0

    storage.resize_(4)
    assert storage.size() == 4
    assert storage.nbytes() == 4
    assert storage.data_ptr() != 0

    storage.resize_(12)
    assert storage.size() == 12
    assert storage.nbytes() == 12
    assert storage.data_ptr() != 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_copy_cpu():
    """
    Feature: storage copy
    Description: Verify the result after coping storage
    Expectation: success
    """
    ms.set_device("CPU")
    a = Tensor(2.0)
    b = a * 1
    c = a * 3
    storage_b = b.untyped_storage()
    storage_c = c.untyped_storage()

    assert b.item() == 2.0

    storage_b.copy_(storage_c)

    assert b.item() == 6.0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_copy_cpu_big_size():
    """
    Feature: storage copy over size
    Description: Verify the result after coping storage
    Expectation: success
    """
    ms.set_device("CPU")
    a = Tensor(np.zeros(1000))
    b = Tensor(np.ones(1000)*3)
    c = Tensor(np.zeros(1000))
    storage_a = a.untyped_storage()
    storage_b = b.untyped_storage()

    storage_b.copy_(storage_a)

    assert b.sum().item() == 0
    assert c.sum().item() == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_storage_copy_after_rng():
    """
    Feature: storage copy
    Description: Verify the result of storage inplace_copy, after setting rng state
    Expectation: success
    """
    ms.set_device("Ascend")
    a = Tensor(2.0)
    b = a * 1
    c = a * 3

    # after setting rng state, the device_target of OpsStatus will become CPU, so reset the device_target to be Ascend
    # in storage inplace_copy if the input tensor's device_type is Ascend.
    state = default_generator.get_state()
    default_generator.set_state(state)

    storage_b = b.untyped_storage()
    storage_c = c.untyped_storage()

    assert b.item() == 2.0

    storage_b.copy_(storage_c)

    assert b.item() == 6.0
