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

import pytest
import mindspore as ms
from mindspore.utils import stress_detect


def test_stress_detect():
    """
    Feature: Stress Detect
    Description: test stress detect on Ascend
    Expectation: stress_detect function return 0.
    """
    ms.set_context(device_target="Ascend")
    a = ms.Tensor(1.0)
    b = ms.Tensor(2.0)
    _ = a * b
    ret = stress_detect()
    assert ret == 0


def test_stress_detect_cpu():
    """
    Feature: Stress Detect
    Description: test stress detect on CPU
    Expectation: stress_detect function will raise exception.
    """
    ms.set_context(device_target="CPU")
    a = ms.Tensor(1.0)
    b = ms.Tensor(2.0)
    _ = a * b
    with pytest.raises(RuntimeError) as e:
        stress_detect()
    assert "Stress detection is not supported" in str(e.value)


def test_stress_detect_gpu():
    """
    Feature: Stress Detect
    Description: test stress detect on GPU
    Expectation: stress_detect function will raise exception.
    """
    ms.set_context(device_target="GPU")
    a = ms.Tensor(1.0)
    b = ms.Tensor(2.0)
    _ = a * b
    with pytest.raises(RuntimeError) as e:
        stress_detect()
    assert "Stress detection is not supported" in str(e.value)
