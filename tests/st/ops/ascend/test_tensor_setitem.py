# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test_tensor_setitem """
import pytest
import numpy as np
from mindspore import Tensor


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_slice_by_bool_broadcast():
    """
    Feature: Tensor-setitem-by-bool support broadcast.
    Description: Tensor-setitem-by-bool support broadcast.
    Expectation: success.
    """
    data_np = np.ones([2, 3, 4], np.float32)
    index_np = np.array([True, False])
    value = 2

    data_tensor = Tensor(data_np)
    index_tensor = Tensor(index_np)

    data_np[index_np] = value
    data_tensor[index_tensor] = value
    assert np.allclose(data_tensor.asnumpy(), data_np)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_slice_by_bool_nan():
    """
    Feature: Tensor-setitem-by-bool support nan.
    Description: Tensor-setitem-by-bool support nan.
    Expectation: success.
    """
    data = Tensor(np.ones([2, 3, 4], np.float32))
    index = Tensor(np.array([False, False]))
    data[index] = Tensor([np.nan])
    assert np.allclose(data.asnumpy(), np.ones([2, 3, 4], np.float32))
