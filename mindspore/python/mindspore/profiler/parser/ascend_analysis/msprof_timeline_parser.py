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
from typing import List

from mindspore.profiler.parser.ascend_analysis.function_event import CANNEvent


class MsprofTimelineParser:
    """Msprof timeline file parser."""

    def __init__(self, msprof_data: List):
        self.timeline_data = msprof_data

    def get_acl_to_npu_data(self):
        """Get all the acl to npu flow events."""
        flow_start_dict, flow_end_dict = {}, {}
        x_event_list, other_event_json = [], []
        for data in self.timeline_data:
            cann_event = CANNEvent(data)
            if cann_event.is_x_event():
                x_event_list.append(cann_event)
            else:
                cann_event_json = cann_event.to_json()
                if cann_event_json:
                    other_event_json.append(cann_event_json)

            if cann_event.is_flow_start_event():
                flow_start_dict[cann_event.id] = (cann_event.tid, cann_event.ts)
            elif cann_event.is_flow_end_event():
                flow_end_dict[(cann_event.pid, cann_event.tid, cann_event.ts)] = cann_event.id
        start_flow_to_npu_dict = defaultdict(list)
        device_data_without_flow_json = []
        for cann_event in x_event_list:
            flow_id = flow_end_dict.get((cann_event.pid, cann_event.tid, cann_event.ts))
            start_flow_info = flow_start_dict.get(flow_id)
            if flow_id is not None and start_flow_info is not None:
                start_flow_to_npu_dict[start_flow_info].append(cann_event)
            else:
                cann_event_json = cann_event.to_json()
                if cann_event_json:
                    device_data_without_flow_json.append(cann_event_json)

        return start_flow_to_npu_dict, device_data_without_flow_json + other_event_json
