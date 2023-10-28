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

import numpy as np
from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException
from mindspore.profiler.parser.base_timeline_generator import BaseTimelineGenerator
from mindspore.profiler.parser.container import TimelineContainer
from mindspore.profiler.parser.cpu_gpu_timeline_generator import CpuTimelineGenerator
from mindspore.profiler.parser.integrator import DeviceTarget
from mindspore.profiler.parser.op_intermediate_parser import OPIntermediateParser


class AscendTimelineGenerator(BaseTimelineGenerator):
    """Generate ascend Timeline data from file."""
    _display_filename = 'ascend_timeline_display_{}.json'
    _timeline_summary_filename = 'ascend_timeline_summary_{}.json'
    _cluster_analyse_filename = 'ascend_cluster_analyse_{}_{}_{}_{}.csv'

    def __init__(self, profiling_dir, device_id, rank_id, rank_size, model):
        super().__init__(DeviceTarget.ASCEND.value, model)
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._rank_id = rank_id
        self._rank_size = rank_size
        self._display_filename = self._display_filename.format(rank_id)
        self._timeline_summary_filename = self._timeline_summary_filename.format(rank_id)

        self.step_time_list_df = np.dtype(
            [('Iteration ID', object), ('Steps', object), ('Iteration Start', float), ('Iteration Time', float)])

        self.aicpu_time_list_dt = np.dtype(
            [('Op Name', object), ('Stream ID', int), ('Task Start Time', float), ('Task Duration', float),
             ('pid', int)])

        self.communication_info_dt = np.dtype(
            [('Op Name', object), ('Stream ID', int), ('Task Start Time', float), ('Task Duration', float),
             ('pid', int)])

    @staticmethod
    def _get_all_reduce_names(communication_info):
        names = []
        for info in communication_info:
            # all_reduce_name format: stream_stream_id_stream_op_index_opname
            all_reduce_name = info[0][info[0].rindex('_') + 1:]
            if all_reduce_name not in names:
                names.append(all_reduce_name)
        return names

    def init_timeline(self, op_summary, steptrace):
        """
        Init timeline metadata, adding all collected info.

        Args:
            op_summary: op data
            steptrace: step data
        """

        logger.info('Initiating timeline...')

        timeline_list = op_summary[~np.isin(op_summary['Task Type'], ['AI_CPU', 'HCCL'])][
            ['Op Name', 'Stream ID', 'Task Start Time', 'Task Duration']]

        timeline_list = timeline_list.tolist()
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._rank_id, self._model)
        cpu_timeline_list = cpu_timeline_generator.get_timeline_data()
        if cpu_timeline_list:
            timeline_list.extend(cpu_timeline_list)
            timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        min_cycle_counter = 0
        if timeline_list:
            min_cycle_counter = timeline_list[0][2]

        # Generate step time.
        self._set_step_start_and_end_op_name(timeline_list)

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

        # Add Scope Name.
        default_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Default")
        gradient_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Gradients")
        recompute_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "recompute_Default")

        # Add AI CPU data into timeline temp list and sort by start time.

        aicpu_op = op_summary[op_summary['Task Type'] == 'AI_CPU']
        if aicpu_op.size:
            aicpu_time_list = np.empty((len(aicpu_op),), dtype=self.aicpu_time_list_dt)
            aicpu_time_list['Op Name'] = aicpu_op['Op Name']
            aicpu_time_list['Stream ID'] = aicpu_op['Stream ID']
            aicpu_time_list['Task Start Time'] = aicpu_op['Task Start Time']
            aicpu_time_list['Task Duration'] = aicpu_op['Task Duration'] + aicpu_op['Task Wait Time']
            aicpu_time_list['pid'] = 9000
            aicpu_time_list = aicpu_time_list.tolist()
            timeline_list.extend(aicpu_time_list)
        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Add AllReduce info to timeline temp list and sort by start time.
        hccl_op = op_summary[op_summary['Task Type'] == 'HCCL']
        if hccl_op.size:
            communication_info = np.empty((len(hccl_op,)), dtype=self.communication_info_dt)
            communication_info['Op Name'] = hccl_op['Op Name']
            communication_info['Stream ID'] = hccl_op['Stream ID']
            communication_info['Task Start Time'] = hccl_op['Task Start Time']
            communication_info['Task Duration'] = hccl_op['Task Duration']
            communication_info['pid'] = 10000
            communication_info = communication_info.tolist()
            communication_info.sort(key=lambda x: float(x[self._start_time_idx]))
        else:
            communication_info = []
        if communication_info:
            logger.debug('AllReduce info found. Start adding info into timeline...')
            cluster_related_timeline = self._get_cluster_timeline(
                timeline_list, communication_info, step_time_list)
            timeline_list.extend(cluster_related_timeline)
            timeline_list.extend(communication_info)
            timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Add step time and scope name info.
        timeline_list.extend(step_time_list)
        timeline_list.extend(default_scope_name_time_list)
        timeline_list.extend(recompute_scope_name_time_list)
        timeline_list.extend(gradient_scope_name_time_list)
        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Init a dict for counting the num of streams.
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, min_cycle_counter)

        # Add format thread meta data.
        self._format_meta_data_list.extend(self._timeline_meta)
        self._timeline_meta = self._format_meta_data_list

        # Update timeline summary info
        timeline_summary = op_summary[np.isin(op_summary['Task Type'], ['AI_CORE', 'AI_CPU', 'HCCL'])][[
            'Op Name', 'Stream ID', 'Task Duration']]
        self._timeline_summary['total_time'] = np.sum(timeline_summary['Task Duration'])
        self._timeline_summary['num_of_streams'] = int(
            len(np.unique(timeline_summary['Stream ID'], return_counts=True)[0]))
        self._timeline_summary['num_of_ops'] = int(len(np.unique(timeline_summary['Op Name'], return_counts=True)[0]))
        self._timeline_summary['op_exe_times'] = int(len(timeline_summary))
        self._timeline_summary['max_scope_name_num'] = int(np.max(
            [len(x) for x in np.char.split(timeline_summary['Op Name'].astype(str), sep='/')]))
        logger.info('Finished adding info into timeline...')

    def init_pynative_timeline(self):
        """Init timeline for pynative model."""
        timeline_list = OPIntermediateParser(self._profiling_dir, self._rank_id).get_timeline_data()
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._rank_id, self._model)
        cpu_timeline_list = cpu_timeline_generator.load_cpu_op_data()
        if cpu_timeline_list:
            self._pynative_clock_synchronize(cpu_timeline_list)
            timeline_list.extend(cpu_timeline_list)

        self._register_op_name(timeline_list)
        self._timeline_summary['op_exe_times'] = len(timeline_list)
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num
        self._timeline_summary['num_of_ops'] = len(self._op_name_list)

        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        min_cycle_counter = float(timeline_list[0][self._start_time_idx])

        step_timeline = self._pynative_get_step_timeline_list(timeline_list)
        timeline_list.extend(step_timeline)

        stream_count_dict = {}
        max_scope_name_num = 0
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, min_cycle_counter)
            self._update_num_of_streams(timeline, stream_count_dict)
            cur_scope_name_num = len(timeline[self._op_name_idx].split('/')) - 1
            max_scope_name_num = max(cur_scope_name_num, max_scope_name_num)

        self._timeline_summary['max_scope_name_num'] = max_scope_name_num
        self._timeline_summary['num_of_streams'] = len(stream_count_dict)

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit from 1ms to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name.split('/')[-1]
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) * factor
        dur = op_meta.duration * factor
        timeline_dict['dur'] = dur
        if op_meta.pid is None:
            timeline_dict['pid'] = int(self._device_id)
            # Update total time of operator execution.
            if op_meta.stream_id not in ["Steps", "Scope Name"]:
                self._timeline_summary['total_time'] += op_meta.duration
        else:  # AllReduce and AI CPU pid
            timeline_dict['pid'] = op_meta.pid

        if op_meta.stream_id == "Scope Name":
            # remove the level of scope name which has a format like "0-conv2-Conv2d".
            timeline_dict['name'] = "-".join(op_meta.op_name.split('-')[1:])
            timeline_dict['scope_level'] = int(op_meta.op_name.split('-')[0])
        elif op_meta.stream_id[:len(self._host_cpu_op_label)] == self._host_cpu_op_label:
            timeline_dict['pid'] = self._HOST_CPU_PID

        self._update_format_meta_data(timeline_dict)
        self._timeline_meta.append(timeline_dict)

    def _get_op_timeline(self, communication_info, source_path):
        """get ai_core and cpu timeline."""
        all_reduce_names = AscendTimelineGenerator._get_all_reduce_names(communication_info)
        timeline_list = OPIntermediateParser(self._profiling_dir, self._rank_id).get_timeline_data(all_reduce_names)
        for timeline in timeline_list:
            timeline[self._tid_idx] = f"Stream #{timeline[self._tid_idx]}"

        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._rank_id, self._model)
        cpu_timeline_list = cpu_timeline_generator.get_timeline_data()
        if cpu_timeline_list:
            self._clock_synchronize_to_device(cpu_timeline_list, source_path)
            timeline_list.extend(cpu_timeline_list)
        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['op_exe_times'] = len(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num
        return timeline_list

    def _clock_synchronize_to_device(self, timeline_list, source_path):
        """Synchronize the timestamp from host to device."""
        host_start_file_path = os.path.join(source_path, f"host_start.log.{self._device_id}")
        dev_start_file_path = os.path.join(source_path, f"dev_start.log.{self._device_id}")
        host_monotonic = 0
        dev_cntvct = 0
        try:
            with open(host_start_file_path) as f_obj:
                lines = f_obj.readlines()
            for line in lines:
                info = line.strip().split(':')
                if len(info) < 2 or info[0] != "clock_monotonic_raw":
                    continue
                host_monotonic = int(info[1])
                break

            with open(dev_start_file_path) as f_obj:
                lines = f_obj.readlines()
            for line in lines:
                info = line.strip().split(':')
                if len(info) < 2 or info[0] != "cntvct":
                    continue
                dev_cntvct = int(info[1])
                break
        except (IOError, OSError) as err:
            logger.critical('Error occurred when read dev_start.log: %s', err)
            raise ProfilerIOException() from err
        if host_monotonic == 0 or dev_cntvct == 0:
            logger.error('Error occurred when read host_monotonic or dev_cntvct time')

        factor = {"factor_ns_to_ms": 1e-6, "factor_ten_ns_to_ns": 10, "factor_ms_to_ns": 1e6}
        for idx, time_item in enumerate(timeline_list):
            host_time = int(float(time_item[self._start_time_idx]) * factor.get("factor_ms_to_ns"))
            device_time = dev_cntvct * factor.get("factor_ten_ns_to_ns") + (host_time - host_monotonic)
            timeline_list[idx][self._start_time_idx] = device_time * factor.get("factor_ns_to_ms")

    def _add_framework_info(self, framework_obj_list):
        """
        Add framework info into timeline metadata.

        Args:
            framework_obj_list (list): The framework metadata.
        """
        logger.debug('Start adding framework info into timeline...')
        # Get the framework info that will be written into timeline.
        framework_info_dict = {}
        for framework_obj in framework_obj_list:
            op_name = framework_obj[0]
            op_type = framework_obj[1]
            op_full_name = framework_obj[4]
            op_info = framework_obj[5]
            framework_info = {
                'name': op_name,
                'args': {
                    'type': op_type,
                    'fullname': op_full_name
                }
            }
            framework_info.get('args').update(op_info)
            framework_info_dict[op_full_name] = framework_info

        # Insert framework info into timeline.
        for timeline_item in self._timeline_meta:
            op_full_name = timeline_item.get('name')
            framework_item = framework_info_dict.get(op_full_name)
            if framework_item:
                timeline_item['name'] = framework_item.get('name')
                timeline_item['args'] = framework_item.get('args')
        logger.debug('Finished adding framework info into timeline...')

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

        # Generate free time that exclude computation and communication time.
        free_timeline = self._get_merged_time_list(
            all_timeline, get_interval_time=True, display_name="free_time"
        )[1]

        self._parse_cluster_metrices(step_info, receive_op_not_overlapped_timeline, comm_not_overlapped_timeline,
                                     collective_comm_not_overlapped_timeline, is_pipeline_parallel)

        res_timeline = []
        res_timeline.extend(comm_not_overlapped_timeline)
        res_timeline.extend(aicore_timeline[2])
        res_timeline.extend(comm_timeline[2])
        res_timeline.extend(free_timeline)

        return res_timeline

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
        per_step_time_list = [0 for i in range(len(step_time_list))]
        step = 0
        step_end_time = step_time_list[step][self._start_time_idx] + \
                        step_time_list[step][self._duration_idx]
        for time_item in metric_timeline:
            start_time = time_item[self._start_time_idx]
            if start_time > step_end_time:
                step += 1
                if step >= len(step_time_list):
                    logger.warning("Compute profiler compute_time_inside_step time, "
                                   "find the data length is more than step count, "
                                   "maybe current graph has multi sub graph, skip the last data.")
                    break
                step_end_time = step_time_list[step][self._start_time_idx] + \
                                step_time_list[step][self._duration_idx]
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

    def _pynative_get_step_timeline_list(self, timeline_list):
        """Get step timeline list for pynative model."""
        step_list = []
        # The timeline starts with the GetNext op
        if len(timeline_list) < 2 or 'GetNext' not in timeline_list[0][self._op_name_idx] and \
                'GetNext' not in timeline_list[1][self._op_name_idx]:
            return step_list
        step = [-1, -1]
        step_num = 0
        tid = "Steps"
        for timeline in timeline_list:
            if 'GetNext' not in timeline[self._op_name_idx]:
                continue
            start_time = float(timeline[self._start_time_idx])
            if step[0] == -1:
                step[0] = start_time
            else:
                step[1] = start_time - step[0]
                step_num = step_num + 1
                step_list.append([str(step_num), tid, float(step[0]), step[1]])
                step = [start_time, -1]
        if step[0] != -1 and step[1] == -1:
            step_num = step_num + 1
            step_list.append([str(step_num), tid, float(step[0]),
                              float(timeline_list[-1][self._start_time_idx]) - step[0]])
        return step_list

    def _pynative_clock_synchronize(self, timeline_list):
        """Synchronize the timestamp from device to host."""
        start_time_file_path = os.path.join(self._profiling_dir, f"start_time_{self._rank_id}.txt")
        try:
            with open(start_time_file_path) as f_obj:
                lines = f_obj.readlines()
                # lines[0] stores the host monotonic time of start training.
                host_monotonic_start_time = int(lines[0].strip().split(':')[-1])
                # lines[1] stores the gpu time of start training.
                gpu_start_time = int(lines[1].strip().split(':')[-1])
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {start_time_file_path}: {err}')
            raise ProfilerIOException() from err
        time_diff = gpu_start_time * 1000 - host_monotonic_start_time
        for idx, time_item in enumerate(timeline_list):
            timeline_list[idx][self._start_time_idx] = int(time_item[self._start_time_idx]) + time_diff
            timeline_list[idx][self._start_time_idx] = timeline_list[idx][self._start_time_idx] / 1000000
            timeline_list[idx][self._duration_idx] = timeline_list[idx][self._duration_idx] / 1000

    def _set_step_start_and_end_op_name(self, timeline_list):
        """Set the start and end operator full name of each step."""
        if not timeline_list or len(timeline_list) < 2:
            return

        start_op_idx = 0
        self._step_end_op_name = timeline_list[-1][self._op_name_idx]
        for i, timeline in enumerate(timeline_list):
            if timeline[self._op_name_idx] == self._step_end_op_name:
                start_op_idx = i + 1
                break

        if start_op_idx >= len(timeline_list):
            start_op_idx = 0
        self._step_start_op_name = timeline_list[start_op_idx][self._op_name_idx]
