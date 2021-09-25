# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Watchpoints."""
from mindspore.offline_debug.debugger_tensor import DebuggerTensor


class WatchpointBase:
    """
    Base class for watchpoints.

    Note:
        - The watchpoint is bounded with tensor names.
        - If multiple checking items is specified for one watch point instance,
          a tensor needs to trigger all of them to trigger the watchpoint.
    """
    @property
    def name(self):
        """Get the name for the watchpoint."""
        raise NotImplementedError

    def check(self):
        """
        Check the watchpoint against the tensors.

        Returns:
            list[WatchpointHit], the hits of the watchpoint.
        """


class WatchpointHit:
    """
    Watchpoint hit.

    Note:
        - This class is not meant to be instantiated by user.
        - The instances of this class is immutable.

    Args:
        tensor (DebuggerTensor): The tensor which hits the watchpoint.
        watchpoint (WatchpointBase): The WatchPointBase object initialized with
            user setting value.
        watchpoint_hit_detail (WatchpointBase): The WatchPointBase object
            initialized with actual value of the Tensor.
        error_code: The code describing error.
    """

    def __init__(self,
                 tensor: DebuggerTensor,
                 watchpoint: WatchpointBase,
                 watchpoint_hit_detail: WatchpointBase,
                 error_code):
        self._tensor = tensor
        self._watchpoint = watchpoint
        self._error_code = error_code
        self._watchpoint_hit_detail = watchpoint_hit_detail

    def __str__(self):
        if self._error_code:
            return f"Watchpoint {self._watchpoint.name} check failed " \
                   f"on tensor {self._tensor.name}. " \
                   f"Error detail: error detail."

        return f"Watchpoint {self._watchpoint.name} triggered on " \
               f"tensor {self._tensor.name}. " \
               f"The setting for watchpoint is mean_gt=0.2, abs_mean_gt=0.3." \
               f"The actual value of the tensor is " \
               f"mean_gt=0.21, abs_mean_gt=0.35."

    @property
    def tensor(self) -> DebuggerTensor:
        """Get the tensor for this watchpoint hit."""
        return self._tensor

    def get_watchpoint(self):
        """Get the original watchpoint."""
        return self._watchpoint

    def get_hit_detail(self):
        """Get the actual values for the thresholds in the watchpoint."""
        return self._watchpoint_hit_detail


class TensorTooLargeWatchpoint(WatchpointBase):
    """
    Tensor too large watchpoint.

    When all specified checking conditions were satisfied, this watchpoint would
    be hit after a check.

    Args:
        tensors (Iterable[DebuggerTensor]): The tensors to check.
        abs_mean_gt (float, optional): The threshold for mean of the absolute
            value of the tensor. When the actual value was greater than this
            threshold, this checking condition would be satisfied.
        max_gt (float, optional): The threshold for maximum of the tensor. When
            the actual value was greater than this threshold, this checking
            condition would be satisfied.
        min_gt (float, optional): The threshold for minimum of the tensor. When
            the actual value was greater than this threshold, this checking
            condition would be satisfied.
        mean_gt (float, optional): The threshold for mean of the tensor. When
            the actual value was greater than this threshold, this checking
            condition would be satisfied.
    """

    def __init__(self, tensors,
                 abs_mean_gt=None, max_gt=None, min_gt=None, mean_gt=None):
        self._tensors = tensors
        self._abs_mean_gt = abs_mean_gt
        self._max_gt = max_gt
        self._min_gt = min_gt
        self._mean_gt = mean_gt

    @property
    def name(self):
        return "TensorTooLarge"
