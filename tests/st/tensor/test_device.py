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
"""Test device"""

import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_get_device():
    """
    Feature: get tensor device
    Description: Verify the result of get device
    Expectation: success
    """
    ms.set_context(device_id=6)
    a = ms.Tensor(1.0)
    assert a.device == "CPU"

    b = a * 1
    assert a.device == "Ascend:6"
    assert b.device == "Ascend:6"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_get_device_cpu():
    """
    Feature: get tensor device
    Description: Verify the result of get device
    Expectation: success
    """
    ms.set_context(device_id=6, device_target="CPU")
    a = ms.Tensor(1.0)
    assert a.device == "CPU"

    b = a * 1
    assert a.device == "CPU"
    assert b.device == "CPU"
