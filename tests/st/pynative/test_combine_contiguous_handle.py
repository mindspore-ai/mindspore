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
import numpy as np
from tests.mark_utils import arg_mark
from mindspore import Tensor
from mindspore.hal.contiguous_tensors_handle import combine_tensor_list_contiguous


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.00)
def test_combine_tensor_list_contiguous_ascend_gpu():
    """
    Feature: test combine tensor list contiguous
    Description: tensor list contiguous mem test
    Expectation: run success
    """
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    z = x + y  # [5, 7, 9]
    e = x + y
    handle = combine_tensor_list_contiguous([z, x, y, e], True)
    assert handle[1].asnumpy() == Tensor(np.array([7]).astype(np.float32))
    assert np.allclose(handle[0: 2].asnumpy(), np.array([5, 7]).astype(np.float32))
    expect_out1 = Tensor(np.zeros(128).astype(np.float32))
    expect_out2 = Tensor(np.zeros(128).astype(np.float32))
    expect_out1[0:3] = expect_out1[0:3] + z
    assert np.allclose(handle[384:].asnumpy(), expect_out1.asnumpy())
    assert np.allclose(handle[:128].asnumpy(), expect_out1.asnumpy())
    expect_out2[1:4] = expect_out2[1:4] + x
    assert np.allclose(handle[127:255].asnumpy(), expect_out2.asnumpy())
    assert np.allclose(handle.slice_by_tensor_index(0, 1).asnumpy(), z.asnumpy())
    assert np.allclose(handle.slice_by_tensor_index(1, 2).asnumpy(), np.array([1, 2, 3]).astype(np.float32))


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_combine_tensor_list_contiguous_cpu():
    """
    Feature: test combine tensor list contiguous
    Description: tensor list contiguous mem test
    Expectation: run success
    """
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    z = x + y  # [5, 7, 9]
    e = x + y
    handle = combine_tensor_list_contiguous([z, x, y, e], True)
    assert handle[1].asnumpy() == Tensor(np.array([7]).astype(np.float32))
    assert np.allclose(handle[0: 2].asnumpy(), np.array([5, 7]).astype(np.float32))
    expect_out1 = Tensor(np.zeros(3).astype(np.float32))
    expect_out1[0:3] = expect_out1[0:3] + z
    assert np.allclose(handle[9:].asnumpy(), expect_out1.asnumpy())
    assert np.allclose(handle[:3].asnumpy(), expect_out1.asnumpy())
    assert np.allclose(handle.slice_by_tensor_index(0, 1).asnumpy(), z.asnumpy())
    assert np.allclose(handle.slice_by_tensor_index(1, 2).asnumpy(), np.array([1, 2, 3]).astype(np.float32))
