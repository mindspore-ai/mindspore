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

"""Test view and inplace for aclop"""

import numpy as np
import pytest
import mindspore as ms
from mindspore import ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_aclop_view_inplace_check():
    """
    Feature: Test pyboost view and inplace
    Description: Test view and inplace with aclop
    Expectation: success
    """
    ms.set_device("Ascend")
    x = ms.from_numpy(np.ones((1024, 1)))

    with pytest.raises(RuntimeError) as e:
        y = x.permute(1,0)
        ops.Assign()(y, ms.mint.zeros_like(y))
        ms.runtime.synchronize()
    assert "Check view and inplace failed" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        y = x[10:]
        ops.Assign()(y, ms.mint.zeros_like(y))
        ms.runtime.synchronize()
    assert "Check view and inplace failed" in str(e.value)
