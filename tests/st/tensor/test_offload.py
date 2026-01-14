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

import pytest
from tests.mark_utils import arg_mark

import mindspore as ms


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_data_sync_after_offload(mode):
    """
    Feature: offload
    Description: support data_sync after tensor is offloaded
    Expectation: no exception
    """
    ms.set_context(mode=mode, pynative_synchronize=True)
    ms.set_device('Ascend')
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    x = x * 2
    x = x.transpose((1, 0))
    y = ms.Parameter(x)
    z = y + 1
    z._offload()
    y._offload()
    y.asnumpy()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_offload_twice(mode):
    """
    Feature: offload
    Description: support invoking offload of tensor repeatedly
    Expectation: no exception
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    x._offload()
    x._offload()
