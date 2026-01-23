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
"""Test Tensor data"""
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from tests.mark_utils import arg_mark


def test_tensor_data_getter_shared_data():
    """
    Feature: Tensor.data
    Description: Test Tensor.data getter with sharing data.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = x.data
    assert x._data_ptr() == y._data_ptr() # pylint:disable=protected-access

def test_tensor_data_getter_inplace():
    """
    Feature: Tensor.data
    Description: Test Tensor.data getter with inplace.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32)).to("Ascend")
    y = x.data
    assert np.allclose(x.asnumpy(), y.asnumpy())

    y[0] = 9
    assert np.allclose(x.asnumpy(), y.asnumpy())

    x[1] = 9
    assert np.allclose(x.asnumpy(), y.asnumpy())

def test_tensor_data_getter_grad_info():
    """
    Feature: Tensor.data
    Description: Test Tensor.data getter with grad info.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    x._requires_grad = True # pylint:disable=protected-access
    y = x.data
    assert not y._requires_grad # pylint:disable=protected-access

def test_tensor_data_getter_tensor_version():
    """
    Feature: Tensor.data
    Description: Test Tensor.data getter with tensor version.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    x[0] = 9
    assert x._version == 1 # pylint:disable=protected-access

    y = x.data
    assert y._version == 0 # pylint:disable=protected-access

def test_tensor_data_setter_shared_data():
    """
    Feature: Tensor.data
    Description: Test Tensor.data setter with sharing data.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = Tensor(np.ones((2, 3), dtype=np.float32) * 2)
    x.data = y
    assert np.allclose(x.asnumpy(), y.asnumpy())

def test_tensor_data_setter_inplace():
    """
    Feature: Tensor.data
    Description: Test Tensor.data setter with inplace.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = Tensor(np.ones((2, 3), dtype=np.float32) * 2).to("Ascend")
    x.data = y
    x[0] = 9
    assert np.allclose(x.asnumpy(), y.asnumpy())

def test_tensor_data_setter_grad_info():
    """
    Feature: Tensor.data
    Description: Test Tensor.data setter with grad.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = Tensor(np.ones((2, 3), dtype=np.float32) * 2)
    x._requires_grad = True # pylint:disable=protected-access
    x.data = y
    assert x._requires_grad # pylint:disable=protected-access

def test_tensor_data_setter_tensor_version():
    """
    Feature: Tensor.data
    Description: Test Tensor.data setter with tensor version.
    Expectation: success
    """
    x = Tensor(np.ones((2, 3), dtype=np.float32))
    y = Tensor(np.ones((2, 3), dtype=np.float32) * 2)
    x[0] = 9
    assert x._version == 1 # pylint:disable=protected-access

    x.data = y
    assert x._version == 1 # pylint:disable=protected-access


def test_tensor_data_setter_with_assign():
    """
    Feature: test tensor data setter
    Description: test set a tensor to a tensor's data
    Expectation: success
    """
    t = ms.Tensor(np.ones((2,2)))
    param = ms.Parameter(np.ones((2,2)))
    assert np.allclose(param.asnumpy(), np.ones((2,2)))
    t.data = ops.normal((2,2), 0, 1)
    param.assign_value(t)
    assert np.allclose(param.asnumpy(), t.asnumpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_tensor_data():
    """
    Feature: Tensor.data
    Description: Test setter/getter of Tensor.data
    Expectation: success
    """

    test_tensor_data_getter_shared_data()
    test_tensor_data_getter_inplace()
    test_tensor_data_getter_grad_info()
    test_tensor_data_getter_tensor_version()

    test_tensor_data_setter_shared_data()
    test_tensor_data_setter_inplace()
    test_tensor_data_setter_grad_info()
    test_tensor_data_setter_tensor_version()
    test_tensor_data_setter_with_assign()
