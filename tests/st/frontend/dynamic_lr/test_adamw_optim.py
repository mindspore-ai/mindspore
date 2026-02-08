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
"""Test dynamic_lr."""
from tests.st.frontend.dynamic_lr.optim_base import OptimFactory


def test_adamw_true_1_1_0():
    """
    Feature: test dynamic lr with AdamW
    Description: test dynamic lr.
    Expectation: the result match with expected result.
    """
    fact = OptimFactory(optim_ex="AdamW", group=True, lr_dynamic=1, if_change=1, if_change_inside=0)
    # 910A降精度运算
    fact.loss = 1e-3
    fact.result_cmp()
