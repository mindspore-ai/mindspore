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
"""Utility helpers for operation testing.

This module provides:
- ReductionOpsFactory: constructs reduction operator testcases and enables
  forward/gradient comparisons across backends.
"""
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._op_info.op_info import OpInfo


class ReductionOpsFactory(OpsFactory):
    """Factory for reduction ops testcases.

    Extends the common factory with net class tweaking and reduction-specific
    sample input builders.
    """
    def __init__(
            self,
            op_info: OpInfo,
            **kwargs,
    ):
        super().__init__(
            op_info,
            **kwargs,
        )
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None
