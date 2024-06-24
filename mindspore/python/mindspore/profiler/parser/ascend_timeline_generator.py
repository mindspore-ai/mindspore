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

import os.path
import glob
import json
import stat
from decimal import Decimal
import numpy as np
from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException
from mindspore.profiler.parser.base_timeline_generator import BaseTimelineGenerator
from mindspore.profiler.parser.integrator import DeviceTarget
from mindspore.profiler.parser.ascend_analysis.fwk_cann_parser import FwkCANNParser
from mindspore.profiler.common.util import get_newest_file


class AscendTimelineGenerator(BaseTimelineGenerator):
    """Generate ascend Timeline data from file."""
    _timeline_display_filename = 'ascend_timeline_display_{}.json'
    _timeline_summary_filename = 'ascend_timeline_summary_{}.json'
    _cluster_analyse_filename = 'ascend_cluster_analyse_{}_{}_{}_{}.csv'
    top_scope_name = ('Default', 'Gradients', 'recompute_Default')
    scope_index = 1
    cpu_index = 2

    def __init__(self, profiling_dir, source_path, mindstudio_profiler_output, rank_id, rank_size, mode,
                 step_list=None):
        super().__init__(DeviceTarget.ASCEND.value, mode)
        self._profiling_dir = profiling_dir
        self._source_path = source_path
        self._mindstudio_profiler_output = mindstudio_profiler_output
        self._rank_id = rank_id
        self._rank_size = rank_size
        self._timeline_display_filename = self._timeline_display_filename.format(rank_id)
        self._timeline_summary_filename = self._timeline_summary_filename.format(rank_id)
        self._timeline_data = []
        self._step_list = step_list

        self.step_time_list_df = np.dtype(
            [('Iteration ID', object), ('Steps', object), ('Iteration Start', float), ('Iteration Time', float)])

        self.aicpu_time_list_dt = np.dtype(
            [('Op Name', object), ('Stream ID', int), ('Task Start Time', float), ('Task Duration', float)])

    def parse_cluster_data(self, op_summary, steptrace):
        """
        Parse cluster data and timeline summary data.

        Args:
            op_summary: op data
            steptrace: step data
        """

        logger.info('parse cluster data...')
        if isinstance(op_summary, np.ndarray) and op_summary.shape[0] == 0 or \
                not isinstance(op_summary, np.ndarray) and not op_summary:
            return
        timeline_list = op_summary[~np.isin(op_summary['Task Type'], ['AI_CPU', 'HCCL'])][
            ['Op Name', 'Stream ID', 'Task Start Time', 'Task Duration']]

        timeline_list = timeline_list.tolist()

        if not isinstance(steptrace, np.ndarray) or steptrace.shape[0] == 0 or not steptrace.tolist():
            iteration_time = op_summary[-1]['Task Start Time'] - op_summary[0]['Task Start Time'] + op_summary[-1][
                'Task Duration'] + op_summary[-1]['Task Wait Time']
            step_time_list = [['1', 'Steps', op_summary[0]['Task Start Time'], iteration_time]]
        else:
            step_time_list = np.empty((len(steptrace),), dtype=self.step_time_list_df)
            step_time_list['Iteration ID'] = \
                np.char.add("Model ID: ",
                            np.char.add(steptrace['Model ID'].astype(str),
                                        np.char.add(" Iteration ID: ",
                                                    steptrace['Iteration ID'].astype(str))))
            step_time_list['Steps'] = 'Steps'
            step_time_list['Iteration Start'] = steptrace['Iteration End'] - steptrace['Iteration Time']
            step_time_list['Iteration Time'] = steptrace['Iteration Time']
            step_time_list = step_time_list.tolist()

        # Add AI CPU data into timeline temp list and sort by start time.
        aicpu_op = op_summary[op_summary['Task Type'] == 'AI_CPU']
        if aicpu_op.size:
            aicpu_time_list = np.empty((len(aicpu_op),), dtype=self.aicpu_time_list_dt)
            aicpu_time_list['Op Name'] = aicpu_op['Op Name']
            aicpu_time_list['Stream ID'] = aicpu_op['Stream ID']
            aicpu_time_list['Task Start Time'] = aicpu_op['Task Start Time']
            aicpu_time_list['Task Duration'] = aicpu_op['Task Duration'] + aicpu_op['Task Wait Time']
            aicpu_time_list = aicpu_time_list.tolist()
            timeline_list.extend(aicpu_time_list)
        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Add AllReduce info to timeline temp list and sort by start time.
        communication_info = op_summary[op_summary['Task Type'] == 'HCCL'][
            ['Op Name', 'Stream ID', 'Task Start Time', 'Task Duration']]
        if communication_info.size:
            communication_info = communication_info.tolist()
            communication_info.sort(key=lambda x: float(x[self._start_time_idx]))
            logger.debug('AllReduce info found. Start adding info into timeline...')
            self._get_cluster_timeline(timeline_list, communication_info, step_time_list)

        # Update timeline summary info
        timeline_summary = op_summary[['Op Name', 'Stream ID', 'Task Duration']]
        self._timeline_summary['total_time'] = np.sum(timeline_summary['Task Duration'])
        self._timeline_summary['num_of_streams'] = int(
            len(np.unique(timeline_summary['Stream ID'], return_counts=True)[0]))
        self._timeline_summary['num_of_ops'] = int(len(np.unique(timeline_summary['Op Name'], return_counts=True)[0]))
        self._timeline_summary['op_exe_times'] = int(len(timeline_summary))
        if self._timeline_summary['op_exe_times'] != 0:
            self._timeline_summary['max_scope_name_num'] = int(np.max(
                [len(x) for x in np.char.split(timeline_summary['Op Name'].astype(str), sep='/')]))
        else:
            self._timeline_summary['max_scope_name_num'] = 0
        logger.info('Finished parse cluster data...')

    def write_timeline_display(self):
        """Write timeline display"""
        logger.info('Writing timeline file...')
        display_file_path = os.path.join(
            self._profiling_dir,
            self._timeline_display_filename
        )
        try:
            with os.fdopen(os.open(display_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as fw:
                json.dump(self._timeline_data, fw, indent=self.indent)
            os.chmod(display_file_path, stat.S_IREAD | stat.S_IWRITE)
            logger.info('Finished file writing!')
        except (IOError, OSError) as err:
            logger.critical('Error occurred when write timeline display file: %s', err)
            raise ProfilerIOException() from err

    def parse_timeline_data(self, pretty=False):
        """
        Get detail timeline
        Returns:
            json, the content of timeline data.
        """
        logger.info("Start parse timeline data...")
        self._pretty = pretty
        timeline_data = []
        all_scope_data = []

        # get msprof data
        msprof_file_name = fr'{self._mindstudio_profiler_output}/msprof_*.json'
        file_list_msprof = glob.glob(msprof_file_name)
        msprof_timeline = []
        if not file_list_msprof:
            logger.error('Could not find msprof_*.json file in %s', self._mindstudio_profiler_output)
        else:
            msprof_timeline = self._parse_msprof_data(get_newest_file(file_list_msprof))

        # get Ascend Hardware for scope
        scope_data = self._parse_ascend_hardware_scope(msprof_timeline)
        all_scope_data.extend(scope_data)

        # get cpu op
        cpu_op_file_name = fr'{self._profiling_dir}/cpu_op_execute_timestamp_{self._rank_id}.txt'
        file_list = glob.glob(cpu_op_file_name)
        if not file_list:
            logger.warning('Could not find cpu op file in %s', self._profiling_dir)
        else:
            cpu_timeline, scope_data = self.parse_cpu_timeline(file_list)
            timeline_data.extend(cpu_timeline)
            all_scope_data.extend(scope_data)

        # parse scope info
        scope_timeline = self._parse_scope_info(all_scope_data)
        timeline_data.extend(scope_timeline)

        oprange_name = self._op_range_name.format(self._rank_id)
        fwk_file_path = fr'{self._profiling_dir}/{self._framework_dir}/{oprange_name}'
        if os.path.exists(fwk_file_path):
            # It is faster not to submit to the pool
            result = self._parse_fwk_device_data(msprof_timeline)
            timeline_data.extend(result.get("trace_data", []))
            self._kernel_events = result.get("kernels", [])
        else:
            timeline_data.extend(msprof_timeline)

        logger.info("All timeline data parse complete.")
        self._timeline_data = timeline_data
        return timeline_data

    def parse_cpu_timeline(self, file_list):
        """Load cpu operator data from file"""
        ms_to_us = 1e3
        ps_to_ns = 1e-3
        new_pid = int(f'{self.cpu_index}{self._rank_id}')
        process_list = [{"name": "process_name",
                         "pid": new_pid,
                         "args": {
                             "name": f"CPU OP Rank{self._rank_id}"
                         },
                         "ph": "M"
                         }, {"name": "process_sort_index", "pid": new_pid,
                             "args": {"sort_index": self.cpu_index}, "ph": "M"}
                        ]
        tid_set = set()
        thread_list = []
        new_timeline = []
        scope_data = []
        try:
            flags = os.O_RDONLY
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    for line in fr:
                        op_list = line.strip().split(';')
                        op_full_name = op_list[0]
                        time_arr = op_list[-1]
                        time_arr = time_arr.split(" ")
                        for time in time_arr:
                            ts, dur, tid = time.split(",")
                            ts = Decimal(ts).quantize(Decimal('0.000')) * Decimal(ps_to_ns).quantize(Decimal('0.000'))

                            if op_full_name and op_full_name.startswith(self.top_scope_name):
                                te = ts + Decimal(dur).quantize(Decimal('0.000'))
                                scope_data.append((op_full_name.split('/')[:-1], ts, te))

                            if int(tid) not in tid_set:
                                tid_set.add(int(tid))
                                thread_list.append({"name": "thread_name",
                                                    "pid": new_pid,
                                                    "tid": int(tid),
                                                    "ph": "M",
                                                    'args': {'name': f'thread {tid}'}
                                                    })

                            new_timeline.append({'name': op_list[0],
                                                 'pid': new_pid,
                                                 'tid': int(tid),
                                                 'ph': 'X',
                                                 'ts': str(ts),
                                                 'dur': float(dur) * ms_to_us,
                                                 'args':
                                                     {'type': op_list[1]}
                                                 })
                break

            return process_list + thread_list + new_timeline, scope_data

        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error('parse_cann_data failed! please theck. detail: %s', err)
            return []

    def _parse_fwk_device_data(self, cann_kernel_data):
        """
        Get framework op range trace data, flow events and hardware kernel events
        """
        fwkcann_parser = FwkCANNParser(self._source_path, cann_kernel_data, self._rank_id, self._step_list)
        fwk_link_data = fwkcann_parser.generate_trace_data()
        kernels = fwkcann_parser.kernels
        result = {"trace_data": fwk_link_data, "kernels": kernels}
        return result

    def _parse_msprof_data(self, file_list):
        """
        parse msprof.json file
        :param file_list:
        :return:
        """
        flags = os.O_RDONLY
        raw_data = []
        try:
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    raw_data.extend(json.load(fr))

            if not raw_data:
                logger.error('Could not found msprof data in file list: %s .', file_list)

            return raw_data

        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error('_parse_msprof_data failed! please theck. detail: %s', err)
            return []

    def _parse_ascend_hardware_scope(self, msprof_timeline):
        """
        parse ascend hardware scope
        """
        scope_data = []
        try:

            for event in msprof_timeline:
                if event.get("ph") == "M":
                    continue

                op_full_name = event.get('name')
                if op_full_name and op_full_name.startswith(self.top_scope_name):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    te = ts + Decimal(event.get('dur')).quantize(Decimal('0.000'))
                    scope_data.append((op_full_name.split('/')[:-1], ts, te))

            return scope_data

        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error('_parse_ascend_hardware_scope failed! please theck. detail: %s', err)
            return []

    def _parse_scope_info(self, scope_data):
        """Parse scope info of op"""
        if not scope_data:
            return []
        new_pid = int(f'{self.scope_index}{self._rank_id}')
        scope_data.sort(key=lambda x: x[1])
        process_list = [
            {"name": "process_name",
             "pid": new_pid,
             "args": {
                 "name": f"Scope Layer Rank{self._rank_id}"
             },
             "ph": "M"},
            {"name": "process_sort_index",
             "pid": new_pid,
             "args": {"sort_index": self.scope_index},
             "ph": "M"}
        ]

        new_events = []
        layer_stack = []
        for layer_name in scope_data[0][0]:
            layer_stack.append([layer_name, scope_data[0][1], scope_data[0][2]])

        for op in scope_data[1:]:
            if op[1] < layer_stack[0][2]:
                # 并行算子只保留前面的
                continue
            flag = True  # 判断上层是否合并， 上层不合并下层也不合并
            for layer_depth, layer_name in enumerate(op[0]):
                if layer_depth >= len(layer_stack):
                    layer_stack.append([layer_name, op[1], op[2]])
                else:
                    if layer_stack[layer_depth][0] == layer_name and flag:
                        layer_stack[layer_depth][2] = op[2]  # 合并
                    else:
                        ts = layer_stack[layer_depth][1]
                        new_events.append({
                            "name": layer_stack[layer_depth][0],
                            "pid": new_pid,
                            "tid": layer_depth,
                            "ph": "X",
                            "ts": str(ts),
                            "dur": float(layer_stack[layer_depth][2] - layer_stack[layer_depth][1])
                        })
                        layer_stack[layer_depth] = [layer_name, op[1], op[2]]
                        flag = False

        thread_list = []
        for index, layer in enumerate(layer_stack):
            thread_list.extend([{
                "name": "thread_name",
                "pid": new_pid,
                "tid": index,
                "args": {
                    "name": f"layer{index}"
                },
                "ph": "M"
            }, {
                "name": "thread_sort_index",
                "pid": new_pid,
                "tid": index,
                "args": {"sort_index": index},
                "ph": "M"
            }])
            if layer:
                ts = layer[1]
                new_events.append({
                    "name": layer[0],
                    "pid": new_pid,
                    "tid": index,
                    "ph": "X",
                    "ts": str(ts),
                    "dur": float(layer[2] - layer[1])
                })

        return process_list + thread_list + new_events

    def _produce_two_separated_timeline(self, timeline, op_name):
        """Produce two separated timeline based on op_name."""
        timeline_include_op_name = []
        timeline_exclude_op_name = []
        for time_item in timeline:
            if op_name in time_item[self._op_name_idx]:
                timeline_include_op_name.append(time_item)
            else:
                timeline_exclude_op_name.append(time_item)
        return timeline_include_op_name, timeline_exclude_op_name

    def _get_cluster_timeline(self, aicore_info, comm_info, step_info):
        """
        Analyse the cluster communication and computation data, and write result to file.

        To analyse the cluster performance bottleneck based on timeline, define the time of a training
        step as "t_total", propose five metrics as follows:
            1) The time that "receive" operators not overlapped by others(t1)
            2) The time that is consumed inside the stage(t_total - t1)
            3) The time that "communication" operators not overlapped by others(t2)
            4) The time that consumed by computation(t_total - t2)
            5) The time that "collective communication" operators not overlapped by others(t3)
        In pipeline parallel mode, we can locate slow stage based on t_total - t1. Inside each stage,
        we can locate slow card based on t_total - t2. The value of t1 indicates the degree that
        communication time between stages slow down the training. The value of t3 indicates the degree
        that communication inside each stage slow down the training.
        """
        is_pipeline_parallel = False
        comm_timeline = self._get_merged_time_list(
            comm_info, display_name="communication"
        )
        aicore_timeline = self._get_merged_time_list(
            aicore_info, get_interval_time=True
        )
        # Consider if the overlap will be 0 or not.
        comm_not_overlapped_timeline = self._get_intersection_time(
            aicore_timeline[0], comm_timeline[0]
        )

        # Process receive part.
        all_timeline = aicore_info + comm_info
        all_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        receive_timeline = self._produce_two_separated_timeline(
            all_timeline, "Receive-op"
        )
        if receive_timeline[0]:
            is_pipeline_parallel = True
        receive_op_merged_timeline = self._get_merged_time_list(receive_timeline[0])[0]
        timeline_exclude_receive_op_interval = self._get_merged_time_list(
            receive_timeline[1], get_interval_time=True
        )[0]
        receive_op_not_overlapped_timeline = self._get_intersection_time(
            timeline_exclude_receive_op_interval, receive_op_merged_timeline
        )

        # Process collective communication part.
        collective_comm_timeline = self._produce_two_separated_timeline(
            comm_info, "Receive-op"
        )[-1]

        collective_comm_not_overlapped_timeline = self._get_intersection_time(
            aicore_timeline[0], self._get_merged_time_list(collective_comm_timeline)[0]
        )

        self._parse_cluster_metrices(step_info, receive_op_not_overlapped_timeline, comm_not_overlapped_timeline,
                                     collective_comm_not_overlapped_timeline, is_pipeline_parallel)

    def _parse_cluster_metrices(self, step_info, receive_op_not_overlapped_timeline, comm_not_overlapped_timeline,
                                collective_comm_not_overlapped_timeline, is_pipeline_parallel):
        """Write the cluster metrices"""
        # Compute these five metrics mentioned above per step.
        recieve_alone_time = self._compute_time_inside_step(receive_op_not_overlapped_timeline, step_info)
        time_info = {"stage_time": [], "computation_time": []}
        comm_alone_time = self._compute_time_inside_step(comm_not_overlapped_timeline, step_info)
        collective_comm_alone_time = self._compute_time_inside_step(
            collective_comm_not_overlapped_timeline, step_info
        )
        step_num = len(step_info)
        for step in range(step_num):
            try:
                if is_pipeline_parallel:
                    time_info.get("stage_time").append(step_info[step][self._duration_idx] - recieve_alone_time[step])
            except IndexError as err:
                logger.error(err)

            try:
                time_info.get("computation_time").append(step_info[step][self._duration_idx] - comm_alone_time[step])
            except IndexError as err:
                logger.error(err)

        metrices_per_step_list = [
            time_info.get("computation_time"), comm_alone_time, time_info.get("stage_time"),
            recieve_alone_time, collective_comm_alone_time
        ]
        if step_num > 1:
            for metric in metrices_per_step_list:
                metric.append(sum(metric[1:]) / (step_num - 1))

        try:
            self._write_cluster_metrices(metrices_per_step_list, is_pipeline_parallel, "Ascend", self._rank_id)
        except (IOError, OSError) as err:
            logger.warning(err)
            raise ProfilerIOException from err

    def _compute_time_inside_step(self, metric_timeline, step_time_list):
        """Compute per step time of metric_timeline."""
        per_step_time_list = [0 for _ in range(len(step_time_list))]
        step = 0
        step_end_time = step_time_list[step][self._start_time_idx] + step_time_list[step][self._duration_idx]
        for time_item in metric_timeline:
            start_time = time_item[self._start_time_idx]
            if start_time > step_end_time:
                step += 1
                if step >= len(step_time_list):
                    logger.warning("Compute profiler compute_time_inside_step time, "
                                   "find the data length is more than step count, "
                                   "maybe current graph has multi sub graph, skip the last data.")
                    break
                step_end_time = step_time_list[step][self._start_time_idx] + step_time_list[step][self._duration_idx]
            per_step_time_list[step] += time_item[self._duration_idx]

        return per_step_time_list

    def _get_intersection_time(self, first_time_list, second_time_list,
                               display_name="communication_not_overlapped"):
        """Get intersection time of two time list."""
        first_list_idx, second_list_idx = 0, 0
        first_list_len = len(first_time_list)
        second_list_len = len(second_time_list)
        intersection_segment_display_list = []

        while first_list_idx < first_list_len and \
                second_list_idx < second_list_len:
            intersection_start = max(
                first_time_list[first_list_idx][self._start_time_idx],
                second_time_list[second_list_idx][self._start_time_idx]
            )
            intersection_end = min(
                first_time_list[first_list_idx][self._duration_idx],
                second_time_list[second_list_idx][self._duration_idx]
            )
            if intersection_start < intersection_end:
                tid = self._tid_dict.get(display_name, [0, 0])
                intersection_segment_display_list.append(
                    [display_name, tid[0],
                     intersection_start, intersection_end - intersection_start, tid[1]]
                )
            if first_time_list[first_list_idx][self._duration_idx] >= \
                    second_time_list[second_list_idx][self._duration_idx]:
                second_list_idx += 1
            else:
                first_list_idx += 1

        return intersection_segment_display_list
