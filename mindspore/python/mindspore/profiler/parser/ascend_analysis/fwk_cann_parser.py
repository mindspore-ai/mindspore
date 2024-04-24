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
"""The parser for integrating data from the framework side and CANN side."""
from collections import defaultdict
from typing import List, Tuple, Optional
from decimal import Decimal

from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.function_event import CANNEvent, MindSporeOpEvent
from mindspore.profiler.parser.ascend_analysis.fwk_file_parser import FwkFileParser
from mindspore.profiler.parser.ascend_analysis.trace_event_manager import TraceEventManager
from mindspore.profiler.parser.ascend_analysis.msprof_timeline_parser import MsprofTimelineParser
from mindspore.profiler.parser.ascend_analysis.profiler_info_parser import ProfilerInfoParser


class FwkCANNParser:
    """The top-level trace view parser."""

    _trace_view_save_path = "trace_view_{}.json"

    def __init__(self, source_path: str, msprof_data: List, rank_id: int):
        source_path = validate_and_normalize_path(source_path)
        ProfilerInfoParser.init_source_path(source_path)
        ProfilerInfoParser.init_rank_id(rank_id)
        fwk_parser = FwkFileParser(source_path, rank_id)
        msprof_timeline_parser = MsprofTimelineParser(msprof_data)
        self._fwk_op_data = fwk_parser.get_op_range_data()
        self._fwk_trace_data = fwk_parser.get_fwk_trace_data(self._fwk_op_data)
        self._acl_to_npu = msprof_timeline_parser.get_acl_to_npu_data()
        self.rank_id: int = rank_id
        self.kernels: List[CANNEvent] = []

    def generate_trace_data(self):
        """
        Generate trace view json timeline file which contains framework side and
        device side data.
        """
        trace_data = self.__link_msop_kernel()
        return self._fwk_trace_data + trace_data

    def __link_msop_kernel(self) -> List:
        """Associate the frame-side operator with the device-side kernel"""
        trace_data = []
        op_data_by_tid = defaultdict(list)
        acl_to_npu_by_tid = {}
        for op_data in self._fwk_op_data:
            op_data_by_tid[op_data.tid].append(op_data)
        for (ts, tid), event_list in self._acl_to_npu.items():
            acl_to_npu_by_tid.setdefault(tid, defaultdict(list))[ts].extend(event_list)
        if op_data_by_tid.keys() != acl_to_npu_by_tid.keys():
            logger.warning("Failed to create link between mindspore operator and kernels.")

        for tid in op_data_by_tid:
            op_idx = 0
            op_data_sorted = sorted(op_data_by_tid[tid], key=lambda x: x.ts)
            acl_sorted = sorted(acl_to_npu_by_tid.get(tid, {}).items(), key=lambda x: x[0])
            for ts, cann_event_list in acl_sorted:
                op_idx, status = FwkCANNParser.__find_launch_op(ts, op_data_sorted, op_idx)
                if not status:
                    continue
                for cann_event in cann_event_list:
                    cann_event.parent = op_data_sorted[op_idx]
                    op_data_sorted[op_idx].children.append(cann_event)
                    flow_list = TraceEventManager.create_mindspore_to_npu_flow(op_data_sorted[op_idx], cann_event)
                    self.kernels.append(cann_event)
                    trace_data += flow_list
        return trace_data

    @staticmethod
    def __find_launch_op(ts: Decimal, op_list: List[MindSporeOpEvent],
                         left: Optional[int] = None, right: Optional[int] = None
                         ) -> Tuple[int, bool]:
        """
        Searching the op_list in [left, right) range and find the operator
        whose start time is larger than ts and end time is less than ts.

        Args:
            ts(Decimal): kernel start time
            op_list(List): MindSporeOpEvent list
            left & right(int): the searching index range is [left, right)

        Return:
            Tuple[int, bool]: the first element is the searched index, the second element
                marks where the operator index is found or not.
        """
        left = 0 if (left is None or left < 0) else left
        right = len(op_list) if (right is None or right < 0) else right
        # The data in range [left, right) is considered.
        while right > left:
            if op_list[left].ts > ts:
                return left, False
            if op_list[left].end_us < ts:
                left += 1
            else:
                return left, True
        return left, False
