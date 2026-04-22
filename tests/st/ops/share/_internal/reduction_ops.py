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
import pytest
from mindspore.graph.api import _pynative_executor
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

    def test_reduction_op_error(self):
        '''
        Test reduction op error cases.
        '''
        if self.op_error_inputs_func is None:
            print(f"\nop_name: {self.op_name} has no op_error_inputs_func, "
                  f"skip test_reduction_op_error.")
            return

        print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_reduction_op_error...")
        for error_input in self.op_error_inputs_func(self.op_info, device=self._device):
            _sample_input = error_input.op_sample_input
            _error_type = error_input.op_error_type

            try:
                self.op(_sample_input.op_input, *_sample_input.op_args, **_sample_input.op_kwargs)
                _pynative_executor.sync()
            except _error_type:
                continue
            else:
                e_type = list(map(lambda x: x.__name__, _error_type if isinstance(_error_type,
                                                                                  (list, tuple)) else [_error_type]))
                print(f"\ntest_binary_op_error catch expect {e_type} failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\nsample_input: {_sample_input.summary()}"
                      f"\nerror_info: {error_input.op_error_info}")
                pytest.fail("没有抛出预期异常")
