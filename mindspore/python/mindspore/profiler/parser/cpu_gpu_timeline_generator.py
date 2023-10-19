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
import csv

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, ProfilerFileNotFoundException, \
    ProfilerParamValueErrorException
from mindspore.profiler.parser.container import TimelineContainer
from mindspore.profiler.parser.base_timeline_generator import BaseTimelineGenerator
from mindspore.profiler.parser.integrator import DeviceTarget
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


class GpuTimelineGenerator(BaseTimelineGenerator):
    """Generate gpu Timeline data from file."""
    _display_filename = 'gpu_timeline_display_{}.json'
    _timeline_summary_filename = 'gpu_timeline_summary_{}.json'
    _output_op_execute_time_file_path = "gpu_op_execute_timestamp_{}.txt"
    _output_activity_execute_time_file_path = "activity_execute_timestamp_{}.txt"
    _output_gpu_activity_info_file_path = "gpu_activity_data_{}.csv"
    _step_trace_original_filename = 'step_trace_profiling_{}.txt'
    _cluster_analyse_filename = 'gpu_cluster_analyse_{}_{}_{}_{}.csv'
    _activity_keys_list = []

    def __init__(self, profiling_dir, device_id, rank_size, model):
        super().__init__(DeviceTarget.GPU.value, model)
        self._device_id = device_id
        self._rank_size = rank_size
        self._profiling_dir = profiling_dir
        self._timeline_meta = []
        self._display_filename = self._display_filename.format(device_id)
        self._timeline_summary_filename = self._timeline_summary_filename.format(device_id)
        self._tid_dict = {
            "receive_op_not_overlapped": (self._RECEIVE_ALONE, self._OP_OVERLAP_PID),
            "exclude_receive_op": (self._ALLREDUCE_ALONE, self._OP_OVERLAP_PID),
            "computation_op": (self._MERGED_COMPUTATION_TID, self._OP_OVERLAP_PID),
            "communication_not_overlapped": (self._PURE_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "communication": (self._MERGED_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "free_time": (self._FREE_TIME_TID, self._OP_OVERLAP_PID)
        }

    def init_timeline(self, reduce_op_type):
        """Init timeline metadata, adding all collected info."""
        timeline_list = self._load_timeline_data(reduce_op_type)

        # Init a dict for counting the num of streams.
        stream_count_dict = {}
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, 0)
            # Updating the collection of streams.
            if len(timeline) == 4:
                self._update_num_of_streams(timeline, stream_count_dict)

        # Add format thread meta data.
        self._format_meta_data_list.extend(self._timeline_meta)
        self._timeline_meta = self._format_meta_data_list

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict)

    def check_op_name(self, op_name):
        """
        Check whether the operator name exists.

        Args:
            op_name (str): The operator name or operator name prefix.

        Returns:
            bool, `True` if the operator name does exist, else `False`.
        """
        if not op_name:
            raise ProfilerParamValueErrorException('The op_name should exist.')
        for op_time_info in self._timeline_meta:
            full_op_name = op_time_info['name']
            if full_op_name and full_op_name.startswith(op_name):
                return True
        return False

    def is_gpu_kernel_async_launch(self):
        """Recognize the solution that launch the gpu kernel async."""
        step_trace_profiling_path = self._get_and_validate_path(
            self._step_trace_original_filename
        )
        try:
            with open(step_trace_profiling_path, 'r') as f_obj:
                line = next(f_obj)
                first_string = line.strip().split()[0]
                # the data format of launch the gpu kernel async is "Default/op1,160123 op-name"
                # otherwise, the data format is "Default/op1 160123,12 "
                return bool(len(first_string.split(',')) == 2)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {step_trace_profiling_path}: {err}')
            raise ProfilerIOException() from err
        except StopIteration:
            logger.warning('No step trace data exists.')
            return False

    def _get_and_validate_path(self, file_name):
        """Generate op or activity file path from file name, and validate this path."""
        file_path = os.path.join(
            self._profiling_dir,
            file_name.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)
        if not os.path.exists(file_path):
            logger.critical(f"Failed to find parsed timeline file {file_path}.")
            raise ProfilerFileNotFoundException('parsed timeline file')

        return file_path

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit of start_time(ts) from 1ns to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name.split('/')[-1]
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) / factor
        dur = op_meta.duration
        timeline_dict['dur'] = dur
        if op_meta.pid is None:
            timeline_dict['pid'] = int(self._device_id)
        else:
            timeline_dict['pid'] = op_meta.pid
        if op_meta.stream_id == "Scope Name":
            # remove the level of scope name which has a format like "0-conv2-Conv2d".
            timeline_dict['name'] = "-".join(op_meta.op_name.split('-')[1:])
            timeline_dict['scope_level'] = int(op_meta.op_name.split('-')[0])
        elif op_meta.stream_id[:len(self._host_cpu_op_label)] == self._host_cpu_op_label:
            timeline_dict['pid'] = self._HOST_CPU_PID

        if len(timeline) > 4:
            # len(timeline) > 4 refers to activity data, else op data.
            # Add args for activity data
            args_dict = {}
            for ix, value in enumerate(timeline[4:]):
                args_dict[self._activity_keys_list[ix]] = value
            timeline_dict['args'] = args_dict
            timeline_dict['tid'] = f"Stream #{timeline_dict.get('tid', '0')}"
        elif op_meta.stream_id not in ["Scope Name", "Steps"]:
            # Update total time of operator execution.
            self._timeline_summary['total_time'] += dur / factor
            self._timeline_summary['op_exe_times'] += 1

        self._update_format_meta_data(timeline_dict)
        self._timeline_meta.append(timeline_dict)

    def _load_timeline_data(self, reduce_op_type):
        """Load timeline data from file."""
        op_file_path = self._get_and_validate_path(
            self._output_op_execute_time_file_path)

        timeline_list, communication_info = self._load_op_data(op_file_path, reduce_op_type)
        communication_info.sort(key=lambda x: float(x[2]))
        # Add host cpu op timeline.
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._device_id, self._model)
        cpu_timeline_list = cpu_timeline_generator.load_cpu_op_data()
        if cpu_timeline_list:
            self._clock_synchronize_to_gpu(cpu_timeline_list)
            timeline_list.extend(cpu_timeline_list)
        timeline_list.sort(key=lambda x: float(x[2]))
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num

        # Generate step time.
        factor_start_time_uint_to_duration = 1e-3
        self._set_step_start_and_end_op_name(timeline_list)
        # Fit gpu kernel async launch solution.
        if self.is_gpu_kernel_async_launch():
            step_time_list = self._get_step_time_list_from_step_trace()
        else:
            step_time_list = self._get_step_time_list(timeline_list, factor_start_time_uint_to_duration)

        # Add Scope Name.
        default_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Default",
                                                                      factor_start_time_uint_to_duration)
        gradient_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Gradients",
                                                                       factor_start_time_uint_to_duration)
        recompute_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "recompute_Default",
                                                                        factor_start_time_uint_to_duration)
        cuda_op_timeline = self._load_activity_data()

        # Add AllReduce info to timeline temp list and sort by start time.
        if communication_info:
            logger.debug('Allreduce info found, Start adding info to timeline...')
            cluster_related_timeline = self._get_cluster_timeline(
                timeline_list, cuda_op_timeline[1], communication_info, step_time_list)
            timeline_list.extend(cluster_related_timeline)
            timeline_list.extend(communication_info)
            timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        timeline_list.extend(default_scope_name_time_list)
        timeline_list.extend(gradient_scope_name_time_list)
        timeline_list.extend(recompute_scope_name_time_list)
        timeline_list.extend(step_time_list)

        timeline_list.sort(key=lambda x: (float(x[self._start_time_idx])))

        # Add cuda activity timeline.
        timeline_list.extend(cuda_op_timeline[0])
        timeline_list.sort(key=lambda x: float(x[2]))

        return timeline_list

    def _clock_synchronize_to_gpu(self, timeline_list):
        """Synchronize the timestamp from device to host."""
        start_time_file_path = os.path.join(self._profiling_dir, f"start_time_{self._device_id}.txt")

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

        time_diff = gpu_start_time - host_monotonic_start_time
        for idx, time_item in enumerate(timeline_list):
            timeline_list[idx][self._start_time_idx] = int(time_item[self._start_time_idx]) + time_diff

    def _load_op_data(self, op_file_path, reduce_op_type):
        """Load operator data from file"""
        op_timeline_list = []
        communication_info = []
        try:
            with open(op_file_path, 'r') as f_obj:
                for line in f_obj:
                    self._timeline_summary['num_of_ops'] += 1
                    op_list = line.strip('\n').strip().split(';')
                    time_arr = op_list[-1]
                    time_arr = time_arr.split(" ")
                    for time in time_arr:
                        time = time.split(",")
                        line_list = op_list[:2] + time
                        communication_op_name = line_list[0].strip().split('/')[-1]
                        if communication_op_name not in reduce_op_type:
                            op_timeline_list.append(line_list)
                        else:
                            communication_info.append(line_list)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when load operator timeline data intermediate file: %s', err)
            raise ProfilerIOException() from err

        return op_timeline_list, communication_info

    def _load_activity_data(self):
        """Load activity data from file"""
        activity_timeline_list = []
        cuda_compute_ops_timeline_list = []
        args_dict = {}
        activity_file_path = self._get_and_validate_path(
            self._output_activity_execute_time_file_path)
        activity_args_file_path = self._get_and_validate_path(
            self._output_gpu_activity_info_file_path)

        if not os.path.exists(activity_args_file_path):
            logger.error(f'The file {activity_args_file_path} does not exist.')
            raise ProfilerFileNotFoundException(activity_args_file_path)
        with open(activity_args_file_path, 'r') as args_file:
            csv_reader = csv.reader(args_file)
            keys_list = next(csv_reader)
            # keys_list format is: name, type, op_full_name, stream_id, block_dim, grid_dim, ...
            self._activity_keys_list = keys_list[1:3] + keys_list[4:6]
            for info in csv_reader:
                args_dict[info[0]] = info[1:3] + info[4:6]

        if not os.path.exists(activity_file_path):
            logger.error(f'The file {activity_file_path} does not exist.')
            raise ProfilerFileNotFoundException(activity_file_path)
        with open(activity_file_path, 'r') as f_obj:
            for line in f_obj:
                line_list = line.strip('\n').split(';')
                # concat activity args info.
                line_list += args_dict.get(line_list[0])
                if not line_list[0].startswith('nccl'):
                    cuda_compute_ops_timeline_list.append(line_list)
                activity_timeline_list.append(line_list)

        return activity_timeline_list, cuda_compute_ops_timeline_list

    def _get_step_time_list_from_step_trace(self):
        """Produce the time of each step based on step_trace_profiling file."""
        # Record the time of each step.
        step_time_list = []
        step_start_op_name = []
        step_end_op_name = []
        step_num = 1
        tid = "Steps"
        step_trace_profiling_path = self._get_and_validate_path(
            self._step_trace_original_filename
        )

        try:
            with open(step_trace_profiling_path, 'r') as f_obj:
                for line in f_obj:
                    line = line.strip().split()
                    step_start_op_name.append(line[0].split(',')[0])
                    step_end_op_name.append(line[3].split(',')[0])
                    cur_step_start_time = float(line[0].split(',')[1])
                    cur_step_end_time = float(line[3].split(',')[1])
                    # convert duration time unit from ns to us.
                    cur_step_duration_time = (cur_step_end_time - cur_step_start_time) / 1e3
                    step_time_item = [str(step_num), tid, cur_step_start_time, cur_step_duration_time]
                    step_time_list.append(step_time_item)
                    step_num += 1
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {step_trace_profiling_path}: {err}')
            raise ProfilerIOException() from err

        return step_time_list

    def _get_cluster_timeline(self, timeline, activity_info, comm_info, step_info):
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
        time_info = {
            "stage_time": [], "computation_time": [], "recieve_alone_time": [], "comm_alone_time": [],
            "collective_comm_alone_time": []
        }
        is_pipeline_parallel = False
        comm_timeline = self._get_merged_time_list(
            comm_info,
            display_name="communication",
            factor=1e-3
        )
        compute_op_timeline = timeline + activity_info
        compute_op_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        compute_timeline = self._get_merged_time_list(
            compute_op_timeline,
            get_interval_time=True,
            factor=1e-3
        )
        # Consider if the overlap will be 0 or not.
        comm_not_overlapped_timeline = self._get_intersection_time(
            compute_timeline[0],
            comm_timeline[0]
        )

        # Process receive part.
        all_timeline = timeline + comm_info
        all_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        receive_op_timeline = self._produce_two_separated_timeline(
            all_timeline,
            "Receive-op"
        )[0]
        if receive_op_timeline:
            is_pipeline_parallel = True
        receive_op_merged_timeline = self._get_merged_time_list(receive_op_timeline,
                                                                factor=1e-3)[0]

        receive_op_not_overlapped_timeline = self._get_intersection_time(
            compute_timeline[0],
            receive_op_merged_timeline,
            display_name="receive_op_not_overlapped"
        )

        # Process collective communication part.
        collective_comm_timeline = self._produce_two_separated_timeline(
            comm_info,
            "Receive-op"
        )[-1]
        collective_comm_merged_timeline = self._get_merged_time_list(collective_comm_timeline,
                                                                     factor=1e-3)[0]
        collective_comm_not_overlapped_timeline = self._get_intersection_time(
            compute_timeline[0],
            collective_comm_merged_timeline,
            display_name="exclude_receive_op"
        )

        # Generate free time that exclude computation and communication time.
        all_timeline = compute_op_timeline + comm_info
        all_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        free_timeline = self._get_merged_time_list(
            all_timeline,
            get_interval_time=True,
            display_name="free_time",
            factor=1e-3
        )[1]

        # Compute these five metrics mentioned above per step.
        time_info["recieve_alone_time"] = self._compute_time_inside_step(receive_op_not_overlapped_timeline, step_info)
        time_info["comm_alone_time"] = self._compute_time_inside_step(comm_not_overlapped_timeline, step_info)
        time_info["collective_comm_alone_time"] = self._compute_time_inside_step(
            collective_comm_not_overlapped_timeline,
            step_info
        )
        step_num = len(step_info)
        for step in range(step_num):
            try:
                if is_pipeline_parallel:
                    time_info.get("stage_time").append(
                        step_info[step][self._duration_idx] - time_info.get("recieve_alone_time")[step]
                    )
            except IndexError as e:
                logger.error(e)
            try:
                time_info.get("computation_time").append(
                    step_info[step][self._duration_idx] - time_info.get("comm_alone_time")[step]
                )
            except IndexError as e:
                logger.error(e)

        metrices_per_step_list = [
            time_info.get("computation_time"), time_info.get("comm_alone_time"),
            time_info.get("stage_time"), time_info.get("recieve_alone_time"),
            time_info.get("collective_comm_alone_time")
        ]
        if step_num > 1:
            for metric in metrices_per_step_list:
                metric.append(sum(metric[1:]) / (step_num - 1))
        try:
            self._write_cluster_metrices(metrices_per_step_list, is_pipeline_parallel, "Gpu", self._device_id)
        except (IOError, OSError) as err:
            logger.warning(err)
            raise ProfilerIOException from err

        res_timeline = []
        res_timeline.extend(comm_not_overlapped_timeline)
        res_timeline.extend(compute_timeline[2])
        res_timeline.extend(comm_timeline[2])
        res_timeline.extend(free_timeline)
        return res_timeline

    def _compute_time_inside_step(self, metric_timeline, step_time_list):
        """Compute per step time of metric_timeline."""
        per_step_time_list = []
        step = 0
        cur_step_metric_time = 0
        factor_us_to_ns = 1e3
        step_end_time = step_time_list[step][self._start_time_idx] + \
                        step_time_list[step][self._duration_idx] * factor_us_to_ns
        for time_item in metric_timeline:
            start_time = time_item[self._start_time_idx]
            if start_time > step_end_time:
                per_step_time_list.append(cur_step_metric_time)
                step += 1
                if step >= len(step_time_list):
                    logger.warning("Compute profiler compute_time_inside_step time, "
                                   "find the data length is more than step count, "
                                   "maybe current graph has multi sub graph, skip the last data.")
                    break
                step_end_time = step_time_list[step][self._start_time_idx] + \
                                step_time_list[step][self._duration_idx] * factor_us_to_ns
                cur_step_metric_time = 0
            cur_step_metric_time += time_item[self._duration_idx]
        per_step_time_list.append(cur_step_metric_time)

        return per_step_time_list

    def _get_intersection_time(self, first_time_list, second_time_list,
                               display_name="communication_not_overlapped"):
        """Get intersection time of two time list."""
        first_list_idx, second_list_idx = 0, 0
        first_list_len = len(first_time_list)
        second_list_len = len(second_time_list)
        intersection_segment_display_list = []
        factor_ns_to_us = 1e-3
        while first_list_idx < first_list_len and second_list_idx < second_list_len:
            intersection_start = max(
                first_time_list[first_list_idx][self._start_time_idx],
                second_time_list[second_list_idx][self._start_time_idx]
            )
            intersection_end = min(
                first_time_list[first_list_idx][self._duration_idx],
                second_time_list[second_list_idx][self._duration_idx]
            )
            if intersection_start < intersection_end:
                intersection_segment_display_list.append(
                    [display_name, self._tid_dict.get(display_name, ('',))[0],
                     intersection_start, (intersection_end - intersection_start) * factor_ns_to_us,
                     self._tid_dict.get(display_name, ('', ''))[1]]
                )
            if first_time_list[first_list_idx][self._duration_idx] >= \
                    second_time_list[second_list_idx][self._duration_idx]:
                second_list_idx += 1
            else:
                first_list_idx += 1

        return intersection_segment_display_list

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


class CpuTimelineGenerator(GpuTimelineGenerator):
    """Generate cpu Timeline data from file."""
    _output_op_execute_time_file_path = "cpu_op_execute_timestamp_{}.txt"
    _display_filename = 'cpu_timeline_display_{}.json'
    _timeline_summary_filename = 'cpu_timeline_summary_{}.json'

    def __init__(self, profiling_dir, device_id, model):
        super().__init__(profiling_dir, device_id, 0, model)
        self._device_target = DeviceTarget.CPU.value

    def get_timeline_data(self):
        """Get timeline data from file."""
        timeline_list = self.load_cpu_op_data()
        factor_ns_to_ms = 1e6
        factor_us_to_ms = 1e3
        for time_item in timeline_list:
            time_item[self._start_time_idx] = float(time_item[self._start_time_idx]) / factor_ns_to_ms
            time_item[self._duration_idx] = float(time_item[self._duration_idx]) / factor_us_to_ms

        return timeline_list

    def init_timeline(self):
        """Init timeline metadata, adding all collected info."""
        timeline_list = self._load_timeline_data()

        # Init a dict for counting the num of streams.
        stream_count_dict = {}
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, 0)
            # Updating the collection of streams.
            if len(timeline) == 4:
                self._update_num_of_streams(timeline, stream_count_dict)

        # Add format thread meta data.
        self._format_meta_data_list.extend(self._timeline_meta)
        self._timeline_meta = self._format_meta_data_list

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict.keys())

    def load_cpu_op_data(self):
        """Load cpu operator data from file"""
        op_file_path = self._get_and_validate_path(
            self._output_op_execute_time_file_path)
        timeline_list = []
        if not os.path.exists(op_file_path):
            logger.info("No cpu operator info.")
            return timeline_list
        timeline_list = self._load_op_data(op_file_path)
        factor_ms_to_us = 1e-3
        for time_item in timeline_list:
            time_item[self._duration_idx] = float(time_item[self._duration_idx]) / factor_ms_to_us

        return timeline_list

    def _get_and_validate_path(self, file_name):
        """Generate op or activity file path from file name, and validate this path."""
        file_path = os.path.join(
            self._profiling_dir,
            file_name.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)

        return file_path

    def _load_op_data(self, op_file_path):
        """Load operator data from file"""
        op_timeline_list = []
        try:
            with open(op_file_path, 'r') as f_obj:
                for line in f_obj:
                    self._timeline_summary['num_of_ops'] += 1
                    op_list = line.strip('\n').strip().split(';')
                    time_arr = op_list[-1]
                    time_arr = time_arr.split(" ")
                    for time in time_arr:
                        time = time.split(",")
                        if len(time) == 3:
                            # for time value is [start_timestamp, duration, tid]
                            # line_list[1] would be like "HostCpuOps" + str(tid)
                            line_list = op_list[:1] + [op_list[1] + str(time[-1])] + time[:-1]
                        else:
                            # for time value is [start_timestamp, duration]
                            line_list = op_list[:2] + time
                        op_timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when load operator timeline data intermediate file: %s', err)
            raise ProfilerIOException() from err

        return op_timeline_list

    def _load_timeline_data(self):
        """Load timeline data from file."""
        timeline_list = self.load_cpu_op_data()

        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num

        # Generate step time.
        factor_start_time_uint_to_duration = 1e-3
        self._set_step_start_and_end_op_name(timeline_list)

        step_time_list = self._get_step_time_list(timeline_list, factor_start_time_uint_to_duration)

        # Add merge compute time and free time
        merge_compute_timeline = self._get_merged_time_list(
            timeline_list, False, "computation_op", factor_start_time_uint_to_duration)[2]
        free_time_timeline = self._get_merged_time_list(
            timeline_list, True, "free_time", factor_start_time_uint_to_duration)[1]

        # Add Scope Name.
        default_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Default",
                                                                      factor_start_time_uint_to_duration)
        gradient_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Gradients",
                                                                       factor_start_time_uint_to_duration)
        recompute_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "recompute_Default",
                                                                        factor_start_time_uint_to_duration)
        timeline_list.extend(default_scope_name_time_list)
        timeline_list.extend(gradient_scope_name_time_list)
        timeline_list.extend(recompute_scope_name_time_list)
        timeline_list.extend(step_time_list)

        timeline_list.sort(key=lambda x: (float(x[self._start_time_idx]), x[self._tid_idx]))
        timeline_list.sort(key=lambda x: float(x[2]))
        timeline_list.extend(merge_compute_timeline)
        timeline_list.extend(free_time_timeline)

        return timeline_list

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit of start_time(ts) from 1ns to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_info = {}
        timeline_info['name'] = op_meta.op_name.split('/')[-1]
        timeline_info['ph'] = 'X'
        timeline_info['tid'] = op_meta.stream_id
        timeline_info['ts'] = (op_meta.start_time - min_cycle_counter) / factor
        dur = op_meta.duration
        timeline_info['dur'] = dur
        timeline_info['pid'] = int(self._device_id)
        if op_meta.stream_id == "Scope Name":
            # remove the level of scope name which has a format like "0-conv2-Conv2d".
            timeline_info['name'] = "-".join(op_meta.op_name.split('-')[1:])
            timeline_info['scope_level'] = int(op_meta.op_name.split('-')[0])
        elif self._host_cpu_op_label == op_meta.stream_id[:len(self._host_cpu_op_label)]:
            timeline_info['pid'] = self._HOST_CPU_PID

        if len(timeline) == 5:
            # len(timeline) == 5 refers to analyse data.
            timeline_info["pid"] = op_meta.pid
        elif op_meta.stream_id not in ["Scope Name", "Steps"]:
            # Update total time of operator execution.
            self._timeline_summary['total_time'] += dur / factor
            self._timeline_summary['op_exe_times'] += 1

        self._update_format_meta_data(timeline_info)
        self._timeline_meta.append(timeline_info)
