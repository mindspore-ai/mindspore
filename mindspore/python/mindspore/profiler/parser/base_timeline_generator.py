# Copyright 2022 Huawei Technologies Co., Ltd
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
"""The integrator for integrating parsed profiling files."""
import os
import stat
import csv
import json

from mindspore import context
from mindspore import log as logger
from mindspore.context import get_auto_parallel_context
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException
from mindspore.profiler.parser.integrator import DeviceTarget
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path

SIZE_LIMIT_DEFAULT = 20 * 1024 * 1024  # 20MB


class BaseTimelineGenerator:
    """
    Analyse timeline data from file.
    """
    # AI Core Op pid is device_id
    _AI_CPU_PID = 9000
    _COMMUNICATION_OP_PID = 10000
    _HOST_CPU_PID = 11000
    _OP_OVERLAP_PID = 12000

    _OP_GPU_ACTIVITY_PID = 13000

    _RECEIVE_ALONE = 7997
    _ALLREDUCE_ALONE = 7998
    _MERGED_COMPUTATION_TID = 7999
    _PURE_COMMUNICATION_TID = 8000
    _MERGED_COMMUNICATION_TID = 8001
    _FREE_TIME_TID = 8002
    _STEPS_TID = 100000
    _SCOPE_NAME_TID = 100001
    _GPU_OP_TID = 100002
    _HOST_CPU_OP_TID = 100003
    _SINGLE_TID = 0

    _STEPS_SORT_INDEX = -4

    _output_timeline_data_file_path = 'output_timeline_data_{}.txt'
    _timeline_meta = []
    _format_meta_data_list = []
    _thread_processed_list = []

    _map_tid_name_to_int = {
        "Steps": (-4, _STEPS_TID),
        "Scope Name": (-3, _SCOPE_NAME_TID),
        "GpuOps": (-2, _GPU_OP_TID),
        "HostCpuOps": (-1, _HOST_CPU_OP_TID)
    }
    _timeline_summary = {
        'total_time': 0,
        'num_of_streams': 0,
        'num_of_ops': 0,
        'op_exe_times': 0,
        'max_scope_name_num': 0,
    }
    _op_name_idx, _tid_idx, _start_time_idx, _duration_idx = 0, 1, 2, 3
    _max_scope_name_num = 0
    _host_cpu_op_label = 'Host CPU OP'
    _gpu_op_label = "GPU Op"
    _ascend_op_label = "Ascend Op"
    _aicore_op_label = "AICORE OP"
    _aicpu_op_label = "AICPU OP"

    _device_id = 0
    _rank_size = 1
    _profiling_dir = ""
    _timeline_summary_filename = ""
    _display_filename = ""
    _op_name_list = []
    _device_target = DeviceTarget.ASCEND.value
    _model = context.GRAPH_MODE

    __col_names__ = ['op_name', 'stream_id', 'start_time', 'duration']

    def __init__(self, device_target, model):
        self._tid_dict = {
            "computation_op": (self._MERGED_COMPUTATION_TID, self._OP_OVERLAP_PID),
            "communication_not_overlapped": (self._PURE_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "communication": (self._MERGED_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "free_time": (self._FREE_TIME_TID, self._OP_OVERLAP_PID)
        }
        self._device_target = str(device_target).lower()
        self._model = model
        self._step_start_op_name = ""
        self._step_end_op_name = ""

    @staticmethod
    def get_parallel_context():
        """Get parallel context."""
        try:
            parallel_mode, stage_num = get_auto_parallel_context("parallel_mode"), get_auto_parallel_context(
                "pipeline_stages")
        except RuntimeError:
            logger.warning("[profiler] the feature of cluster bottleneck analyse "
                           "is not supported in offline parse mode.")
            parallel_mode = "data_parallel"
            stage_num = 1
        if stage_num > 1:
            parallel_mode = "pipeline-parallel"
        elif parallel_mode != "data_parallel":
            parallel_mode = "model-parallel"
        else:
            parallel_mode = "data-parallel"
        return parallel_mode, stage_num

    @staticmethod
    def _update_num_of_streams(timeline, stream_count_dict):
        """Update number of streams."""
        stream_id = timeline[1]
        if stream_id in ["Steps", "Scope Name"]:
            return
        if stream_id not in stream_count_dict.keys():
            stream_count_dict[stream_id] = 1
        else:
            stream_count_dict[stream_id] += 1

    def get_thread_label_name(self):
        """Get process and thread config."""
        device_process_label = self._get_device_process_label()
        return [
            {"name": "process_labels", "ph": "M", "pid": self._device_id, "args": {"labels": device_process_label}},
            {"name": "process_labels", "ph": "M", "pid": self._AI_CPU_PID, "args": {"labels": self._aicpu_op_label}},
            {"name": "process_labels", "ph": "M", "pid": self._COMMUNICATION_OP_PID,
             "args": {"labels": "Communication Op"}},
            {"name": "process_labels", "ph": "M", "pid": self._HOST_CPU_PID,
             "args": {"labels": self._host_cpu_op_label}},
            {"name": "process_labels", "ph": "M", "pid": self._OP_OVERLAP_PID,
             "args": {"labels": "Op Overlap Analyse"}},
            {"name": "process_labels", "ph": "M", "pid": self._OP_GPU_ACTIVITY_PID,
             "args": {"labels": "Activity Op"}},

            {"name": "process_sort_index", "ph": "M", "pid": self._device_id, "args": {"sort_index": 0}},
            {"name": "process_sort_index", "ph": "M", "pid": self._AI_CPU_PID, "args": {"sort_index": 10}},
            {"name": "process_sort_index", "ph": "M", "pid": self._COMMUNICATION_OP_PID, "args": {"sort_index": 20}},
            {"name": "process_sort_index", "ph": "M", "pid": self._HOST_CPU_PID, "args": {"sort_index": 30}},
            {"name": "process_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "args": {"sort_index": 40}},

            {"name": "thread_name", "ph": "M", "pid": self._HOST_CPU_PID, "tid": self._HOST_CPU_OP_TID,
             "args": {"name": "Host CPU Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMPUTATION_TID,
             "args": {"name": "Merged Computation Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._PURE_COMMUNICATION_TID,
             "args": {"name": "Pure Communication Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMMUNICATION_TID,
             "args": {"name": "Merged Communication Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._FREE_TIME_TID,
             "args": {"name": "Free Time"}},
            {"name": "thread_name", "ph": "M", "pid": self._device_id, "tid": self._STEPS_TID,
             "args": {"name": "Steps"}},
            {"name": "thread_name", "ph": "M", "pid": self._device_id, "tid": self._SINGLE_TID,
             "args": {"name": "Ops"}},

            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMPUTATION_TID,
             "args": {"sort_index": self._MERGED_COMPUTATION_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._PURE_COMMUNICATION_TID,
             "args": {"sort_index": self._PURE_COMMUNICATION_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMMUNICATION_TID,
             "args": {"sort_index": self._MERGED_COMMUNICATION_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._FREE_TIME_TID,
             "args": {"sort_index": self._FREE_TIME_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._device_id, "tid": self._STEPS_TID,
             "args": {"sort_index": self._STEPS_SORT_INDEX}},
        ]

    def write_timeline(self, size_limit=SIZE_LIMIT_DEFAULT):
        """Load data according to the parsed profiling files."""
        # Write timeline to file.
        logger.info('Writing timeline file...')
        timeline_meta = self.write_timeline_to_json_by_limitation(size_limit)
        logger.info('Finished file writing!')
        return timeline_meta

    def write_timeline_to_json_by_limitation(self, size_limit):
        """Write timeline to json by limitation."""
        display_file_path = os.path.join(
            self._profiling_dir,
            self._display_filename
        )
        display_file_path = validate_and_normalize_path(display_file_path)

        try:
            with os.fdopen(os.open(display_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as json_file:
                json_file.write('[')
                for _, item in enumerate(self._timeline_meta):
                    json.dump(item, json_file)
                    if "scope_level" in item.keys():
                        self._max_scope_name_num = max(
                            self._max_scope_name_num, item["scope_level"] + 1)
                    file_size = os.path.getsize(display_file_path)
                    json_file.write(',')
                    if file_size > size_limit:
                        break
                label_name_json = json.dumps(self.get_thread_label_name())
                label_name_json = label_name_json.lstrip('[')
                json_file.write(label_name_json)
                os.chmod(display_file_path, stat.S_IREAD | stat.S_IWRITE)
            return self._timeline_meta
        except (IOError, OSError) as err:
            logger.critical('Error occurred when write timeline display file: %s', err)
            raise ProfilerIOException() from err

    def write_timeline_summary(self):
        """Write timeline summary to json."""
        timeline_summary_file_path = os.path.join(
            self._profiling_dir,
            self._timeline_summary_filename
        )

        timeline_summary_file_path = validate_and_normalize_path(timeline_summary_file_path)

        try:
            with os.fdopen(os.open(timeline_summary_file_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as json_file:
                json.dump(self._timeline_summary, json_file)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when write timeline summary file: %s', err)
            raise ProfilerIOException() from err
        if os.path.exists(timeline_summary_file_path):
            os.chmod(timeline_summary_file_path, stat.S_IREAD | stat.S_IWRITE)

    def _get_device_process_label(self):
        """Get device process label."""
        device_process_label = self._aicore_op_label
        if self._device_target == DeviceTarget.ASCEND.value:
            if self._model == context.GRAPH_MODE:
                device_process_label = self._aicore_op_label
            elif self._model == context.PYNATIVE_MODE:
                device_process_label = self._ascend_op_label
        elif self._device_target == DeviceTarget.GPU.value:
            device_process_label = self._gpu_op_label
        elif self._device_target == DeviceTarget.CPU.value:
            device_process_label = self._host_cpu_op_label
        return device_process_label

    def _get_merged_time_list(self, time_list, get_interval_time=False, display_name="computation_op", factor=1):
        """
        Get merged time segment list.

        The process of merge is, for example, there is a list [[1,5], [2,6], [7,8]],
        each items in this list contains a start_time and end_time,
        the merged result is [[1,6], [7,8]].
        """
        time_merged_segment_list = []
        tid = self._tid_dict.get(display_name, (0, 0))[0]
        pid = self._tid_dict.get(display_name, (0, 0))[1]
        for time_item in time_list:
            time_segment = list(map(float, time_item[self._start_time_idx:self._duration_idx + 1]))
            time_segment[1] = time_segment[0] + time_segment[1] / factor
            if not time_merged_segment_list or \
                    time_segment[0] > time_merged_segment_list[-1]:
                time_merged_segment_list.extend(time_segment)
            else:
                time_merged_segment_list[-1] = max(
                    time_merged_segment_list[-1],
                    time_segment[1]
                )

        # merged_display_list data used for ui page.
        merged_display_list = []
        for i in range(len(time_merged_segment_list) // 2):
            merged_display_list.append([display_name, tid, time_merged_segment_list[i * 2],
                                        (time_merged_segment_list[i * 2 + 1] - time_merged_segment_list[
                                            i * 2]) * factor, pid])

        if get_interval_time:
            time_merged_segment_list = time_merged_segment_list[1:-1]

        # merged_res_list data used to compute overlap with other time_list.
        merged_res_list = []
        for i in range(len(time_merged_segment_list) // 2):
            merged_res_list.append(
                [display_name, tid, time_merged_segment_list[i * 2], time_merged_segment_list[i * 2 + 1], pid])

        # interval_display_list is interval time used for ui page.
        interval_display_list = []
        for i in range(len(time_merged_segment_list) // 2):
            interval_display_list.append([display_name, tid, time_merged_segment_list[i * 2],
                                          (time_merged_segment_list[i * 2 + 1] - time_merged_segment_list[
                                              i * 2]) * factor, pid])

        return merged_res_list, interval_display_list, merged_display_list

    def _update_format_meta_data(self, timeline_dict):
        """Update format meta data which control the display arrange and map the thread name."""
        thread_name_meta_data = {
            "name": "thread_name",
            "pid": int(self._device_id),
            "tid": 100000,
            "ts": 0,
            "ph": "M",
            "cat": "__metadata",
            "args": {
                "name": "Steps"
            }
        }
        tid_name = timeline_dict['tid']
        sort_index = 0

        if tid_name in self._map_tid_name_to_int:
            sort_index, tid = self._map_tid_name_to_int.get(tid_name)
        elif tid_name.startswith("Stream"):
            tid = int(tid_name.split("#")[-1])
            sort_index = tid
        else:
            return

        if self._host_cpu_op_label == tid_name[:len(self._host_cpu_op_label)]:
            thread_name_meta_data['pid'] = self._HOST_CPU_PID

        thread_name_meta_data["tid"] = tid
        thread_name_meta_data.get("args")["name"] = tid_name
        self._format_meta_data_list.append(thread_name_meta_data)

        thread_name_meta_data['name'] = "thread_sort_index"
        thread_name_meta_data["args"] = {"sort_index": sort_index}
        self._format_meta_data_list.append(thread_name_meta_data)
        timeline_dict["tid"] = tid

        if tid_name in self._thread_processed_list:
            return
        self._thread_processed_list.append(tid_name)

    def _get_max_scope_name_num(self, timeline_list):
        """Get the max number of scope level from all operator."""
        max_scope_name_num = 0
        for time_item in timeline_list:
            cur_scope_name_num = len(time_item[self._op_name_idx].split('/')) - 1
            max_scope_name_num = max(cur_scope_name_num, max_scope_name_num)

        return max_scope_name_num

    def _get_scope_name_time_list(self, timeline_list, subgraph, factor_start_time_to_duration=1):
        """Produce the timeline of hierarchical scope name."""
        # the key of scope_name_start_duration_dict is scope name, the value is a dict which store the
        # start and end index of time_item in timeline_list.
        scope_name_start_duration_dict = {}
        scope_name_time_list = []
        sort_idx = {"op_full_name_idx": 0, "scope_name_idx": 0, "invalid_idx": -1}
        for idx, time_item in enumerate(timeline_list):
            scope_name_list = time_item[sort_idx.get("op_full_name_idx")].split('/')[:-1]
            # skip Default/InitDataSetQueue operator.
            if time_item[sort_idx.get("op_full_name_idx")].startswith("Default/InitDataSetQueue"):
                scope_name_list = []
            # process scope name of subgraph(Default/Gradients/recompute_Default) only.
            if scope_name_list and scope_name_list[0] != subgraph:
                scope_name_list = []
            # add the level of scope name, used to distinguish the same name at different scope level.
            scope_name_list = [f"{scope_level}-{scope_name}"
                               for scope_level, scope_name in enumerate(scope_name_list)]

            # update the start and end index of time_item according to current scope_name
            for scope_name in scope_name_list:
                if scope_name not in scope_name_start_duration_dict:
                    scope_name_start_duration_dict[scope_name] = {'start_item_idx': idx, 'end_item_idx': idx}
                if scope_name_start_duration_dict.get(scope_name)['start_item_idx'] == sort_idx.get("invalid_idx"):
                    scope_name_start_duration_dict[scope_name] = {'start_item_idx': idx, 'end_item_idx': idx}
                else:
                    scope_name_start_duration_dict.get(scope_name)['end_item_idx'] = idx
            # if the key(scope name) in scope_name_start_duration_dict does not appear in scope_name_list,
            # it means this key(scope name) is end and it is append to scope_name_time_list.
            for key, val in scope_name_start_duration_dict.items():
                if val['start_item_idx'] == sort_idx.get("invalid_idx"):
                    continue
                if (key not in scope_name_list) \
                        or idx == (len(timeline_list) - 1) \
                        or time_item[sort_idx.get("op_full_name_idx")] == self._step_end_op_name:
                    start_time = timeline_list[val['start_item_idx']][self._start_time_idx]
                    duration = (float(timeline_list[val['end_item_idx']][self._start_time_idx]) - float(start_time)) * \
                               factor_start_time_to_duration + \
                               float(timeline_list[val['end_item_idx']][self._duration_idx])
                    scope_name_time_list.append([key, "Scope Name", start_time, duration])
                    scope_name_start_duration_dict.get(key)['start_item_idx'] = sort_idx.get("invalid_idx")

        # x[scope_name_idx] is a scope name like "0-Default".
        # if two element in scope_name_time_list have the same start time,
        # the previous element in list will displayed at the higher line in UI page.
        scope_name_time_list.sort(
            key=lambda x: (float(x[self._start_time_idx]), int(x[sort_idx.get("scope_name_idx")].split('-')[0]))
        )

        return scope_name_time_list

    def _set_step_start_and_end_op_name(self, timeline_list):
        """Set the start and end operator full name of each step."""
        if not timeline_list:
            return
        start_op_idx = 0
        if timeline_list[0][self._op_name_idx].startswith("Default/InitDataSetQueue"):
            start_op_idx = 1
        self._step_start_op_name = timeline_list[start_op_idx][self._op_name_idx]
        self._step_end_op_name = self._step_start_op_name
        if len(timeline_list) > (start_op_idx + 1):
            for time_item in timeline_list[start_op_idx + 1:]:
                if time_item[self._op_name_idx] != self._step_start_op_name:
                    self._step_end_op_name = time_item[self._op_name_idx]
                else:
                    break

    def _get_step_time_list(self, timeline_list, factor_start_time_to_duration=1):
        """Produce the time of each step."""
        # Record the time of each step.
        step_time_list = []
        step_num = 1
        tid = "Steps"
        cur_step_start_time, cur_step_duration_time = 0, 0
        for time_item in timeline_list:
            if time_item[self._op_name_idx] == self._step_start_op_name:
                cur_step_start_time = time_item[self._start_time_idx]
            if time_item[self._op_name_idx] == self._step_end_op_name:
                cur_step_duration_time = (float(time_item[self._start_time_idx]) - float(cur_step_start_time)) * \
                                         float(factor_start_time_to_duration) + float(time_item[self._duration_idx])
                step_time_item = [str(step_num), tid, float(cur_step_start_time), cur_step_duration_time]
                step_time_list.append(step_time_item)
                step_num += 1

        return step_time_list

    def _write_cluster_metrices(self, metrices, is_pipeline_parallel, device_target, dev_id):
        """Write cluster metric."""
        # Note that the feature of cluster bottleneck analyse is not supported in offline parse mode,
        # due to that parallel context is not set.
        if context.get_context("mode") == context.PYNATIVE_MODE:
            return
        parallel_mode, stage_num = BaseTimelineGenerator.get_parallel_context()

        unit = 1 if device_target == "Ascend" else 1e3
        time_decimal_digits = 4
        cluster_analyse_file_path = os.path.join(
            self._profiling_dir,
            self._cluster_analyse_filename.format(parallel_mode, stage_num, self._rank_size, dev_id)
        )
        cluster_analyse_file_path = validate_and_normalize_path(cluster_analyse_file_path)

        with os.fdopen(os.open(cluster_analyse_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                       'w') as file_handle:
            csv_writer = csv.writer(file_handle)
            if is_pipeline_parallel:
                header = [
                    'computation_time', 'communication_alone_time', 'stage_time',
                    'receive_alone_time', 'collective_communication_alone_time'
                ]
                zip_metrices = zip(metrices[0], metrices[1], metrices[2], metrices[3], metrices[4])
            else:
                header = ['computation_time', 'communication_alone_time']
                zip_metrices = zip(metrices[0], metrices[1])
            csv_writer.writerow(header)
            for row_data in zip_metrices:
                row_data = [round(val / unit, time_decimal_digits) for val in row_data]
                csv_writer.writerow(row_data)
        os.chmod(cluster_analyse_file_path, stat.S_IREAD | stat.S_IWRITE)

    def _register_op_name(self, timeline_list):
        """Register op name to op name list."""
        for timeline in timeline_list:
            if timeline and timeline[self._op_name_idx] not in self._op_name_list:
                self._op_name_list.append(timeline[self._op_name_idx])
