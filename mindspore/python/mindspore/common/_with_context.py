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

"""With context for call"""
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import MixedPrecisionType


class _PyNativeCellCall:
    """PyNative cell call context."""

    def __init__(self, obj, args, kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs
        self.output = None
        # Prevent bprop processed for multiple times
        self._is_call_new_graph = False

    def __enter__(self):
        # Set grad flag and create new top cell is not exist
        if self.obj.requires_grad:
            self._is_call_new_graph = True
            _pynative_executor.set_grad_flag(True)
            _pynative_executor.new_graph(self.obj, *self.args, **self.kwargs)
        elif self.obj.get_inputs() is not None:
            _pynative_executor.set_eval_use_dynamic_shape_process(True)

        # bprop cell in middle
        if self.obj.has_bprop and not self._is_call_new_graph and _pynative_executor.grad_flag():
            _pynative_executor.new_graph(self.obj, *self.args, **self.kwargs)

        # Set mixed precision
        if self.obj.mixed_precision_type is not None:
            _pynative_executor.set_mixed_precision_type(self.obj.mixed_precision_type)

        # For cell distinguish HookBackward op
        if self.obj.enable_backward_hook:
            _pynative_executor.set_hook_id(self.obj)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            _pynative_executor.clear_res()
            return False

        # Reset
        if self.obj.requires_grad:
            _pynative_executor.end_graph(self.obj, self.output, *self.args, **self.kwargs)
        elif self.obj.get_inputs() is not None:
            _pynative_executor.set_eval_use_dynamic_shape_process(False)

        # bprop cell in middle
        if self.obj.has_bprop and not self._is_call_new_graph and _pynative_executor.grad_flag():
            _pynative_executor.end_graph(self.obj, self.output, *self.args, **self.kwargs)

        # mixed precision reset
        if self.obj.mixed_precision_type is not None:
            _pynative_executor.set_mixed_precision_type(MixedPrecisionType.NOTSET, False)

        # Reset cell id to 'None'
        if self.obj.enable_backward_hook:
            _pynative_executor.set_hook_id()

        self._is_call_new_graph = False
        return True
