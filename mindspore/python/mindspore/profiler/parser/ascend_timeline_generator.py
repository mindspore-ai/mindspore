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
import re
import stat
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException
from mindspore.profiler.parser.base_timeline_generator import BaseTimelineGenerator
from mindspore.profiler.parser.container import TimelineContainer
from mindspore.profiler.parser.cpu_gpu_timeline_generator import CpuTimelineGenerator
from mindspore.profiler.parser.integrator import DeviceTarget
from mindspore.profiler.parser.ascend_analysis.fwk_cann_parser import FwkCANNParser


class AscendTimelineGeneratorOld(BaseTimelineGenerator):
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
            communication_info = np.empty((len(hccl_op),), dtype=self.communication_info_dt)
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
        logger.info('Finished adding info into timeline...')

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


def get_newest_file(file_list):
    new_file_list = {}
    for file_path in file_list:
        key = '_'.join(file_path.split('.')[0].split('/')[-1].split('_')[:-1])
        if key not in new_file_list or new_file_list[key] < file_path:
            new_file_list[key] = file_path
    return list(new_file_list.values())


class AscendTimelineGenerator(BaseTimelineGenerator):
    """Generate ascend Timeline data from file."""
    _timeline_display_filename = 'ascend_timeline_display_{}.json'
    _timeline_summary_filename = 'ascend_timeline_summary_{}.json'
    _cluster_analyse_filename = 'ascend_cluster_analyse_{}_{}_{}_{}.csv'
    top_scope_name = ('Default', 'Gradients', 'recompute_Default')
    step_trace_index = 1
    cann_index = 2
    scope_index = 3
    ascend_hardware_index = 4
    hccl_index = 5
    cpu_index = 6
    overlap_index = 7

    def __init__(self, profiling_dir, source_path, rank_id, mode):
        super().__init__(DeviceTarget.ASCEND.value, mode)
        self._profiling_dir = profiling_dir
        self._source_path = source_path
        self._msprof_timeline_dir = os.path.join(source_path, 'timeline')
        self._rank_id = rank_id
        self._timeline_display_filename = self._timeline_display_filename.format(rank_id)
        self._timeline_summary_filename = self._timeline_summary_filename.format(rank_id)
        self._timeline_data = []

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
                json.dump(self._timeline_data, fw)
            os.chmod(display_file_path, stat.S_IREAD | stat.S_IWRITE)
            logger.info('Finished file writing!')
        except (IOError, OSError) as err:
            logger.critical('Error occurred when write timeline display file: %s', err)
            raise ProfilerIOException() from err

    def parse_timeline_data(self):
        """
        Get detail timeline
        Returns:
            json, the content of timeline data.
        """
        logger.info("Start parse timeline data...")

        timeline_data = []
        task_list = []
        hardware_data_list = []
        cann_data_list = []
        hccl_data_list = []

        with ThreadPoolExecutor() as pool:

            all_scope_data = []  # 所有带scope的算子

            # get step_trace data
            step_trace_file_name = fr'{self._msprof_timeline_dir}/step_trace_*.json'
            file_list_step_trace = glob.glob(step_trace_file_name)
            if not file_list_step_trace:
                logger.error('Could not find step trace file in %s', self._msprof_timeline_dir)
            else:
                task_list.append(pool.submit(self._parse_step_trace_data, get_newest_file(file_list_step_trace)))

            # get overlap analysis
            overlap_file_name = fr'{self._msprof_timeline_dir}/msprof_*.json'
            file_list = glob.glob(overlap_file_name)
            if not file_list:
                logger.error('Could not find overlap analysis file in %s', self._msprof_timeline_dir)
            else:
                task_list.append(pool.submit(self._parse_overlap_analysis_data, get_newest_file(file_list)))

            # get cpu op
            cpu_op_file_name = fr'{self._profiling_dir}/cpu_op_execute_timestamp_{self._rank_id}.txt'
            file_list = glob.glob(cpu_op_file_name)
            if not file_list:
                logger.warning('Could not find cpu op file in %s', self._profiling_dir)
            else:
                cpu_timeline, scope_data = self.parse_cpu_timeline(file_list)
                timeline_data.extend(cpu_timeline)
                all_scope_data.extend(scope_data)

            # get Ascend Hardware
            hardware_file_name = fr'{self._msprof_timeline_dir}/task_time_*.json'
            file_list_hardware = glob.glob(hardware_file_name)
            if not file_list_hardware:
                logger.error('Could not find ascend hardware file in %s', self._msprof_timeline_dir)
            else:
                ascend_timeline, scope_data = self._parse_ascend_hardware_data(get_newest_file(file_list_hardware))
                timeline_data.extend(ascend_timeline)
                hardware_data_list.extend(ascend_timeline)
                all_scope_data.extend(scope_data)

            # parse scope info
            task_list.append(pool.submit(self._parse_scope_info, all_scope_data))

            # get hccl
            hccl_file_name = fr'{self._msprof_timeline_dir}/hccl_*.json'
            file_list_hccl = glob.glob(hccl_file_name)
            if not file_list_hccl:
                logger.error('Could not find hccl file in %s', self._msprof_timeline_dir)
            else:
                hccl_data = self._parse_hccl_data(get_newest_file(file_list_hccl))
                timeline_data.extend(hccl_data)
                hccl_data_list.extend(hccl_data)

            # get CANN
            cann_file_name = fr'{self._msprof_timeline_dir}/msprof_*.json'
            file_list = glob.glob(cann_file_name)
            if not file_list:
                logger.error('Could not find overlap analysis file in %s', self._msprof_timeline_dir)
            else:
                cann_data = self._parse_cann_data(get_newest_file(file_list))
                timeline_data.extend(cann_data)
                cann_data_list.extend(cann_data)

            oprange_name = self._op_range_name.format(self._rank_id)
            fwk_file_path = fr'{self._profiling_dir}/{self._framework_dir}/{oprange_name}'
            if os.path.exists(fwk_file_path):
                # It is faster not to submit to the pool
                msprof_side_data = hardware_data_list + cann_data_list + hccl_data_list
                result = self._parse_fwk_device_data(msprof_side_data)
                timeline_data.extend(result.get("trace_data", []))
                self._kernel_events = result.get("kernels", [])

            self._wait_task_and_update(task_list, timeline_data)

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
            print('parse_cann_data failed! please theck. detail: %s', err)
            return []

    def _wait_task_and_update(self, task_list: list, timeline_data: list):
        """
        Wait the tasks to finish and get result
        """
        all_done = list(range(len(task_list)))
        while all_done:
            for ind, t in enumerate(task_list):
                if ind in all_done and t.done():
                    timeline_data.extend(t.result())
                    all_done.remove(ind)

    def _parse_fwk_device_data(self, cann_kernel_data):
        """
        Get framework op range trace data, flow events and hardware kernel events
        """
        fwkcann_parser = FwkCANNParser(self._source_path, cann_kernel_data, self._rank_id)
        fwk_link_data = fwkcann_parser.generate_trace_data()
        kernels = fwkcann_parser.kernels
        result = {"trace_data": fwk_link_data, "kernels": kernels}
        return result

    def _parse_step_trace_metadata(self, raw_data):
        """
        Get step trace by merge models
        """
        pattern1 = re.compile(r'Step Trace\(Model ID:(\d)+\)')
        pattern2 = re.compile(r'(\d)+')
        tid_mapper = {}
        pid = None
        for event in raw_data:
            if event.get("ph") != "M":
                continue

            if event.get('name') == 'process_name':
                pid = event.get('pid')
                continue

            if event.get('name') == 'thread_name':
                arg_name = event.get('args', {}).get('name')
                arg_name = re.search(pattern1, arg_name)
                if not arg_name:
                    continue
                model_id = re.search(pattern2, arg_name.group())
                if not model_id:
                    continue
                model_id = model_id.group()
                tid = event.get('tid')
                tid_mapper[tid] = f'Model {model_id}'

        return tid_mapper, pid

    def _parse_step_trace_data(self, file_list):
        """
        parse step trace data
        """
        try:
            flags = os.O_RDONLY
            raw_data = []
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    raw_data.extend(json.load(fr))

            tid_mapper, pid = self._parse_step_trace_metadata(raw_data)
            if not pid:
                logger.error('Could not found process_name pid. method: _parse_step_trace_data')
                return []

            new_pid = int(f'{self.step_trace_index}{self._rank_id}')

            new_events = []
            for event in raw_data:
                if event.get('pid') != pid:
                    continue
                if event.get('name') == 'process_name' and event.get('ph') == 'M':
                    event['args']['name'] = f"Step Trace Rank{self._rank_id}"
                elif event.get('name') == 'process_sort_index' and event.get('ph') == 'M':
                    event['args']['sort_index'] = self.step_trace_index
                else:
                    arg_name = tid_mapper.get(event.get('tid'))
                    if not arg_name:
                        continue

                event['pid'] = new_pid
                if event.get('ts'):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    event['ts'] = str(ts)
                new_events.append(event)
            return new_events

        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error('parse_step_trace_data failed! please theck. detail: %s', err)
            return []

    def _parse_overlap_analysis_data(self, file_list):
        """
        parse overlap analysis data
        """
        try:
            flags = os.O_RDONLY
            raw_data = []
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    raw_data.extend(json.load(fr))

            pid = None
            for event in raw_data:
                if event.get('name') == 'process_name' and event.get("ph") == "M" and \
                        event.get('args').get('name') == 'Overlap Analysis':
                    pid = event.get('pid')
                    break

            if not pid:
                print('Could not found process_name pid. method: _parse_overlap_analysis_data')
                return []

            new_events = []
            new_pid = int(f'{self.overlap_index}{self._rank_id}')
            for event in raw_data:
                if event.get('pid') != pid:
                    continue

                if event.get('name') == 'process_name' and event.get("ph") == "M":
                    event["args"]["name"] += f" Rank{self._rank_id}"

                if event.get('name') == 'process_sort_index' and event.get("ph") == "M":
                    event["args"]["sort_index"] = self.overlap_index

                event['pid'] = new_pid
                if event.get('ts'):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    event['ts'] = str(ts)

                new_events.append(event)

            return new_events

        except (IOError, OSError, json.JSONDecodeError) as err:
            print('parse_overlap_analysis_data failed! please theck. detail: %s', err)
            return []

    def _parse_ascend_hardware_data(self, file_list):
        """
        parse ascend hardware data
        """
        flags = os.O_RDONLY
        raw_data = []

        new_events = []
        pid = None
        tid_mapper = {}
        tid_set = set()
        new_pid = int(f'{self.ascend_hardware_index}{self._rank_id}')
        new_metadata = [{
            "name": "process_name",
            "pid": new_pid,
            "args": {
                "name": f"Ascend Hardware Rank{self._rank_id}"
            },
            "ph": "M"
        }, {"name": "process_sort_index", "pid": new_pid,
            "args": {"sort_index": self.ascend_hardware_index}, "ph": "M"}]
        scope_data = []
        try:
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    raw_data.extend(json.load(fr))

            for event in raw_data:
                if event.get('name') == 'process_name' and event.get("ph") == "M" and \
                        event.get('args').get('name') == 'Ascend Hardware':
                    pid = event.get('pid')

                elif event.get('name') == 'thread_name' and event.get("ph") == "M" and \
                        'Stream' in event.get('args').get('name'):
                    event['pid'] = new_pid
                    tid_mapper.update({event.get('tid'): event})
            if not pid:
                logger.error('Could not found process_name pid. method: _parse_ascend_hardware_data')
                return []

            for event in raw_data:
                if event.get("ph") == "M":
                    continue

                op_full_name = event.get('name')
                if op_full_name and op_full_name.startswith(self.top_scope_name):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    te = ts + Decimal(event.get('dur')).quantize(Decimal('0.000'))
                    scope_data.append((op_full_name.split('/')[:-1], ts, te))

                if event.get('ts'):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    event['ts'] = str(ts)
                event['pid'] = new_pid
                tid_set.add(event.get('tid'))
                new_events.append(event)

            for tid in tid_set:
                thread_event = tid_mapper.get(tid)
                if thread_event is None:
                    thread_event = {"name": "thread_name", "pid": new_pid,
                                    "tid": tid, "args": {"name": f"Stream {tid}"}, "ph": "M"}
                new_metadata.append(thread_event)
            return new_metadata + new_events, scope_data

        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error('parse_ascend_hardware_data failed! please theck. detail: %s', err)
            return []

    def _parse_hccl_data(self, file_list):
        """
        parse hccl data
        """
        try:
            flags = os.O_RDONLY
            raw_data = []
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    raw_data.extend(json.load(fr))

            pid = None
            tid_mapper = {}
            tid_set = set()
            new_events = []
            new_pid = int(f'{self.hccl_index}{self._rank_id}')
            model_id_set = set()

            for event in raw_data:
                if event.get('name') == 'process_name' and event.get("ph") == "M" and \
                        event.get('args').get('name') == 'HCCL':
                    pid = event.get('pid')

                elif event.get('name') == 'thread_name' and event.get("ph") == "M" and \
                        ('Plane' in event.get('args').get('name') or 'Communication' in event.get('args').get('name')):
                    event['pid'] = new_pid
                    tid_mapper.update({event.get('tid'): event})

            if not pid:
                logger.error('Could not found process_name pid. method: _parse_hccl_data')
                return []

            for event in raw_data:
                if event.get("ph") == "M":
                    continue

                model_id = event.get('args', {}).get('model id')
                model_id_set.add(model_id)
                tid = event.get('tid')

                if event.get('ts'):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    event['ts'] = str(ts)

                event['pid'] = new_pid
                tid_set.add(tid)
                new_events.append(event)

            new_metadata = [{
                "name": "process_name",
                "pid": new_pid,
                "args": {
                    "name": f"HCCL Rank{self._rank_id}"
                },
                "ph": "M"
            }, {"name": "process_sort_index", "pid": new_pid,
                "args": {"sort_index": self.hccl_index}, "ph": "M"}]

            for tid in tid_set:
                new_metadata.append(tid_mapper.get(tid))
            return new_metadata + new_events

        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error('parse_hccl_data failed! please theck. detail: %s', err)
            return []

    def _parse_cann_data(self, file_list):
        """
        Parse cann dataa.
        """
        try:
            flags = os.O_RDONLY
            raw_data = []
            for file_path in file_list:
                with os.fdopen(os.open(file_path, flags, 0o400), 'r') as fr:
                    raw_data.extend(json.load(fr))

            pid = None
            for event in raw_data:
                if event.get('name') == 'process_name' and event.get("ph") == "M" and \
                        event.get('args').get('name') == 'CANN':
                    pid = event.get('pid')
                    break

            if not pid:
                print('Could not found process_name pid. method: _parse_cann_data')
                return []

            new_events = []
            new_pid = int(f'{self.cann_index}{self._rank_id}')
            for event in raw_data:
                if event.get('pid') != pid:
                    continue
                if event.get('name') == 'process_name' and event.get("ph") == "M":
                    event["args"]["name"] += f" Rank{self._rank_id}"

                if event.get('name') == 'process_sort_index' and event.get("ph") == "M":
                    event["args"]["sort_index"] = self.cann_index

                event['pid'] = new_pid
                if event.get('ts'):
                    ts = Decimal(event.get('ts')).quantize(Decimal('0.000'))
                    event['ts'] = str(ts)

                new_events.append(event)

            return new_events

        except (IOError, OSError, json.JSONDecodeError) as err:
            print('parse_cann_data failed! please theck. detail: %s', err)
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
                # print(op[0])
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
