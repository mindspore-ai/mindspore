# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Guard the retained common predicate after removing ops.is_tensor."""
import numpy as np
import pytest
import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('kind', ['tensor', 'parameter', 'numpy', 'none', 'scalar', 'list', 'tuple', 'class'])
def test_retained_common_is_tensor(kind):
    """
    Feature: Retained mindspore.common.is_tensor predicate.
    Description: Check Tensor subclasses and representative non-Tensor objects.
    Expectation: Only Tensor and Parameter instances return True, with a Python bool result.
    """
    values = {'tensor': ms.Tensor([1], ms.float32),
              'parameter': ms.Parameter(ms.Tensor([1], ms.float32), name='value'),
              'numpy': np.array([1]), 'none': None, 'scalar': 1,
              'list': [1], 'tuple': (1,), 'class': ms.Tensor}
    result = ms.common.is_tensor(values[kind])
    assert result is (kind in ('tensor', 'parameter'))
