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
""" test tensor pin_memory """

import numpy as np
import pytest
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import mint, Tensor
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def construct(self, x):
        return x.pin_memory()

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_pin_memory(mode):
    """
    Feature: copy tensor to pin memory.
    Description: test tensor.pin_memory() with a cpu tensor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    assert not x.is_pinned()
    net = Net()
    y = net(x)
    assert y.is_pinned()
    np.allclose(y, x, rtol=1e-5, equal_nan=True)

    # tensor y is on pinned memory, then z is also on pinned memory, and is same with y
    z = net(y)
    assert z.is_pinned()
    assert z.storage().data_ptr() == y.storage().data_ptr()

    with pytest.raises(RuntimeError):
        x_npu = x.move_to('Ascend')
        net(x_npu)

    x_share_mem = x._shared_host_memory_with_device_() # pylint: disable=W0212
    assert not x_share_mem.is_pinned()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_pin_memory_after_empty_like(mode):
    """
    Feature: copy tensor to pin memory.
    Description: test tensor.pin_memory() after mint.empty_like().
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    t = mint.rand((100, 100))
    assert not t.is_pinned()
    t1 = mint.empty_like(t, device='CPU').pin_memory()
    assert t1.is_pinned()
    assert t1.shape == t.shape
