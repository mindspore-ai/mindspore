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
"""Json trace event manager"""

from typing import Dict, List

from mindspore.profiler.parser.ascend_analysis.constant import Constant
from mindspore.profiler.parser.ascend_analysis.function_event import BaseEvent


class TraceEventManager:
    """Chrome trace format json object manager."""

    @classmethod
    def create_x_event(cls, event: BaseEvent, cat: str) -> Dict:
        """Create a X event."""
        x_event = {
            "ph": "X", "name": event.name, "pid": event.pid, "tid": event.tid,
            "ts": str(event.ts), "dur": str(event.dur), "cat": cat, "args": event.args
        }
        return x_event

    @classmethod
    def create_m_event(cls, pid: int, tid_list: set) -> List:
        """Create some metadata event."""
        # framework sidee trace information display format: MindSpore(pid pid_value): CPU
        event_list = [
            # process information
            {"ph": "M", "name": Constant.PROCESS_NAME, "pid": pid, "tid": 0, "args": {"name": "MindSpore"}},
            {"ph": "M", "name": Constant.PROCESS_LABEL, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
            {"ph": "M", "name": Constant.PROCESS_SORT, "pid": pid, "tid": 0, "args": {"sort_index": 0}},
        ]
        for tid in tid_list:
            sort_index = tid
            event_list.extend(
                [{"ph": "M", "name": Constant.THREAD_NAME, "pid": pid, "tid": tid, "args": {"name": f"Thread {tid}"}},
                 {"ph": "M", "name": Constant.THREAD_SORT, "pid": pid, "tid": tid, "args": {"sort_index": sort_index}}])
        return event_list

    @classmethod
    def create_mindspore_to_npu_flow(cls, start_event: BaseEvent, end_event: BaseEvent) -> List:
        """Create flow events link mindspore operator and npu kernel."""
        flow_id = str(end_event.ts)
        return [{"ph": "s", "bp": "e", "name": "mindspore_to_npu", "id": flow_id, "pid": start_event.pid,
                 "tid": start_event.tid, "ts": str(start_event.ts), "cat": "async_npu"},
                {"ph": "f", "bp": "e", "name": "mindspore_to_npu", "id": flow_id, "pid": end_event.pid,
                 "tid": end_event.tid, "ts": str(end_event.ts), "cat": "async_npu"}]

    @classmethod
    def create_mindspore_to_self_flow(cls, start_event: BaseEvent, end_event: BaseEvent) -> List:
        """Create flow events link mindspore operator and npu kernel."""
        flow_id = start_event.flow_id
        return [{"ph": "s", "bp": "e", "name": "mindspore_to_self", "id": flow_id, "pid": start_event.pid,
                 "tid": start_event.tid, "ts": str(start_event.ts), "cat": "async_mindspore"},
                {"ph": "f", "bp": "e", "name": "mindspore_to_self", "id": flow_id, "pid": end_event.pid,
                 "tid": end_event.tid, "ts": str(end_event.ts), "cat": "async_mindspore"}]
