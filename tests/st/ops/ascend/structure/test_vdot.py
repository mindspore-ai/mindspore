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
""" test vdot"""
from tests.mark_utils import arg_mark
import numpy as np
import mindspore as ms


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_vdot_basic():
    """
    Feature: Structured ops.
    Description: Test vdot ops.
    Expectation: Success.
    """
    a = ms.Tensor([2, 3], dtype=ms.float32)
    b = ms.Tensor([2, 1], dtype=ms.float32)
    assert np.allclose(ms.mint.vdot(a, b), [7.0], 1e-5, 1e-5)
