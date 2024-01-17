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
"""msprof timeline file parser"""
from collections import defaultdict
from typing import List, Dict, Tuple

from mindspore.profiler.parser.ascend_analysis.function_event import CANNEvent


class MsprofTimelineParser:
    """Msprof timeline file parser."""

    ACL_TO_NPU = "acl_to_npu"
    START_FLOW = "s"
    END_FLOW = "f"
    TIMELINE = "timeline"

    def __init__(self, msprof_data: List):
        self.timeline_data = msprof_data

    def get_acl_to_npu_data(self) -> Dict[Tuple, List[CANNEvent]]:
        """Get all the acl to npu flow events."""
        flow_start_dict, flow_end_dict = {}, {}
        cann_event_list: List[CANNEvent] = []
        for data in self.timeline_data:
            cann_event = CANNEvent(data)
            cann_event_list.append(cann_event)
            if cann_event.is_flow_start_event():
                flow_start_dict[cann_event.id] = (cann_event.ts, cann_event.tid)
            elif cann_event.is_flow_end_event():
                flow_end_dict[cann_event.unique_id] = cann_event.id
        acl_to_npu_dict = defaultdict(list)
        for cann_event in cann_event_list:
            if not cann_event.is_x_event():
                continue
            corr_id = flow_end_dict.get(cann_event.unique_id)
            acl_ts_tid = flow_start_dict.get(corr_id)
            if corr_id is not None and acl_ts_tid is not None:
                acl_to_npu_dict[acl_ts_tid].append(cann_event)
        return acl_to_npu_dict
