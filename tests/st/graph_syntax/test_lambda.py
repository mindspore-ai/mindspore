# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test map for lambda with fv. """
import os
import pytest
import mindspore as ms

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_map_lambda_with_fv():
    """
    Feature: Support map for lambda with FV.
    Description: Support map for lambda with FV.
    Expectation: No exception.
    """
    os.environ['MS_DEV_PRE_LIFT'] = '1'
    @ms.jit()
    def map_lambda_with_fv(x, y, z):
        number_add = lambda x, y: x + y + z
        return map(number_add, (x,), (y,))

    res = map_lambda_with_fv(1, 5, 9)
    del os.environ['MS_DEV_PRE_LIFT']
    assert res == (15,)
