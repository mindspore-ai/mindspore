# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import csv
import json
import os
import stat
from decimal import Decimal

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, \
    ProfilerFileNotFoundException, ProfilerRawFileException, ProfilerParamValueErrorException
from mindspore.profiler.common.util import query_latest_trace_time_file, to_int, to_millisecond
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.container import TimelineContainer

SIZE_LIMIT_DEFAULT = 20 * 1024 * 1024  # 20MB


class Integrator:
    """
    The integrator for integrating parsed profiling files.

    Args:
        profiling_dir (str): The directory where the parsed profiling files are
            located.
        device_id (str): The device ID.
    """
    _file_name_aicore_detail_time = 'output_op_compute_time_{}.txt'
    _file_name_aicpu_time = 'output_data_preprocess_aicpu_{}.txt'
    _file_name_framework = 'framework_raw_{}.csv'
    _header_aicore_type = ['op_type', 'execution_time', 'execution_frequency',
                           'percent']
    _header_aicore_detail = ['full_op_name', 'execution_time']
    _header_aicpu = ['serial_number', 'op_type', 'total_time', 'dispatch_time',
                     'run_start', 'run_end']

    _file_name_aicore_type_time = 'aicore_intermediate_{}_type.csv'
    _file_name_aicore_detail_info = 'aicore_intermediate_{}_detail.csv'
    _col_names_detail = ['op_name', 'op_type', 'avg_execution_time', 'subgraph', 'full_op_name', 'op_info']
    _none_filter_condition_key = ['is_display_detail', 'is_display_full_op_name']
    _none_sort_col_names = ['op_info']
    _aicore_data = []
    _aicore_detail_data = []
    _aicore_trace_data = []

    def __init__(self, profiling_dir, device_id):
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._op_time_cache = {}
        self._total_time = Decimal('0.0')

    def integrate(self):
        """Integrate the parsed profiling files."""
        self._parse_aicore_detail_time()
        self._parse_aicore_type_time()
        self._parse_aicpu_time()

    def get_aicore_data(self):
        self._aicore_data_load()
        return self._aicore_data

    def get_aicore_detail_data(self):
        self._aicore_detail_data_load()
        return self._aicore_detail_data

    def get_aicore_trace_data(self):
        self._aicore_trace_data_load()
        return self._aicore_trace_data

    def query_for_all_reduce(self):
        return self._query_for_all_reduce()

    def query_and_sort_by_op_type(self, filter_condition, op_type_order):
        return self._query_and_sort_by_op_type(filter_condition, op_type_order)

    def _parse_aicore_type_time(self):
        """Parse the parsed AICORE operator type file."""
        framework_file = os.path.join(
            self._profiling_dir,
            self._file_name_framework.format(self._device_id)
        )
        framework_file = validate_and_normalize_path(framework_file)
        if not os.path.isfile(framework_file):
            return

        op_name_type_cache = {}
        with open(framework_file, 'r') as src_file:
            csv_reader = csv.reader(src_file)
            _ = next(csv_reader)

            for row in csv_reader:
                op_name_type_cache[row[3]] = row[5]

        op_type_time_cache = {}
        for full_op_name, op_time in self._op_time_cache.items():
            op_type = op_name_type_cache.get(full_op_name)
            if op_type_time_cache.get(op_type) is None:
                op_type_time_cache[op_type] = [op_time, 1]
            else:
                op_type_time_cache[op_type][0] += op_time
                op_type_time_cache[op_type][1] += 1

        op_type_file_name = 'aicore_intermediate_' + self._device_id + '_type.csv'
        op_type_file_path = os.path.join(self._profiling_dir, op_type_file_name)
        with open(op_type_file_path, 'w') as type_file:
            csv_writer = csv.writer(type_file)
            csv_writer.writerow(self._header_aicore_type)

            for op_type, op_type_time_info in op_type_time_cache.items():
                type_info = [
                    op_type, op_type_time_info[0], op_type_time_info[1],
                    round((op_type_time_info[0] / self._total_time) * 100, 2)
                ]
                csv_writer.writerow(type_info)

    def _parse_aicore_detail_time(self):
        """Parse the parsed AICORE operator time file."""
        aicore_detail_file = os.path.join(
            self._profiling_dir,
            self._file_name_aicore_detail_time.format(self._device_id)
        )
        aicore_detail_file = validate_and_normalize_path(aicore_detail_file)
        if not os.path.isfile(aicore_detail_file):
            return

        op_detail_file_name = 'aicore_intermediate_' + self._device_id + '_detail.csv'
        op_detail_file_path = os.path.join(
            self._profiling_dir, op_detail_file_name
        )
        with open(aicore_detail_file, 'r') as src_file:
            row = src_file.readline()
            if row.startswith('op_name'):
                _ = src_file.readline()
            elif row.startswith('====='):
                _ = src_file.readline()
                _ = src_file.readline()
            else:
                return

            with open(op_detail_file_path, 'w') as detail_file:
                csv_writer = csv.writer(detail_file)
                csv_writer.writerow(self._header_aicore_detail)

                while True:
                    row = src_file.readline()
                    if not row:
                        break

                    op_infos = row.split()
                    if op_infos[0] == 'total':
                        self._total_time = Decimal(op_infos[2])
                        continue
                    self._op_time_cache[op_infos[0]] = Decimal(op_infos[1])
                    csv_writer.writerow([op_infos[0], op_infos[1]])

    def _parse_aicpu_time(self):
        """Parse the parsed AICPU operator time file."""
        aicpu_file = os.path.join(
            self._profiling_dir,
            self._file_name_aicpu_time.format(self._device_id)
        )
        aicpu_file = validate_and_normalize_path(aicpu_file)
        if not os.path.isfile(aicpu_file):
            return

        save_file_name = 'aicpu_intermediate_' + self._device_id + '.csv'
        save_file_path = os.path.join(self._profiling_dir, save_file_name)
        with open(aicpu_file, 'r') as src_file:
            row = src_file.readline()
            if not row.startswith('serial_number'):
                return
            with open(save_file_path, 'w') as save_file:
                csv_writer = csv.writer(save_file)
                csv_writer.writerow(self._header_aicpu)

                while True:
                    row = src_file.readline()
                    if not row:
                        break
                    infos = row.split()
                    if infos[0] == 'AI':
                        continue
                    csv_writer.writerow(infos)

    def _aicore_data_load(self):
        """Load data according to the parsed AICORE operator types file."""
        op_type_file_path = os.path.join(
            self._profiling_dir,
            self._file_name_aicore_type_time.format(self._device_id)
        )
        op_type_file_path = validate_and_normalize_path(op_type_file_path)
        if not os.path.isfile(op_type_file_path):
            logger.warning('The file <%s> does not exist.', op_type_file_path)
            return

        with open(op_type_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for info in csv_reader:
                self._aicore_data.append([info[0], float(info[1]), int(info[2]), float(info[3])])

    def _aicore_detail_data_load(self):
        """Load data according to the parsed AICORE operator file."""
        op_detail_file_path = os.path.join(
            self._profiling_dir,
            self._file_name_aicore_detail_info.format(self._device_id)
        )
        framework_file_path = os.path.join(
            self._profiling_dir,
            self._file_name_framework.format(self._device_id)
        )
        op_detail_file_path = validate_and_normalize_path(op_detail_file_path)
        framework_file_path = validate_and_normalize_path(framework_file_path)
        if not os.path.isfile(op_detail_file_path):
            logger.warning('The file <%s> does not exist.', op_detail_file_path)
            return
        if not os.path.isfile(framework_file_path):
            logger.warning('The file <%s> does not exist.', framework_file_path)
            return

        framework_infos = dict()
        with open(framework_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for info in csv_reader:
                framework_infos[info[3]] = [
                    info[3], info[4], info[5], info[6], json.loads(info[7]) if info[7] else None]

        with open(op_detail_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for info in csv_reader:
                framework_info = framework_infos.get(info[0])
                self._aicore_detail_data.append(
                    [
                        framework_info[1], framework_info[2], float(info[1]),
                        framework_info[3], framework_info[0], framework_info[4]
                    ]
                )
        del framework_infos

    def _aicore_trace_data_load(self):
        """Load data according to the parsed AICORE operator types file."""
        file_path = query_latest_trace_time_file(self._profiling_dir, int(self._device_id))
        if not file_path:
            logger.error("Failed to find parsed trace time file.")
            raise ProfilerFileNotFoundException('parsed step trace time file')
        file_path = validate_and_normalize_path(file_path)
        with open(file_path, 'r') as handle:
            csv_reader = csv.reader(handle)
            self.__column__ = next(csv_reader)
            self._aicore_trace_data = list(csv_reader)
        self._size = len(self._aicore_trace_data) - 1
        self._load_point_info()

    def _load_point_info(self):
        """Load point info."""
        file_path = os.path.join(self._profiling_dir, 'step_trace_point_info.json')
        file_path = validate_and_normalize_path(file_path)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    self._point_info = json.load(file)
                except (json.JSONDecodeError, TypeError) as err:
                    logger.warning(err)
                    raise ProfilerRawFileException('Fail to parse point info file.')

    def _query_for_all_reduce(self):
        """
        Query for all reduce info.

        Returns:
            list[dict], reduce information. Each item is the reduce info for one step.
            The reduce info is format like:
            {stream_id: List[Tuple(start_point, end_point, duration, field_name)]}.
        """
        self._aicore_trace_data_load()
        reduce_infos = []
        for row_info in self._aicore_trace_data[:-1]:
            row_info_dict = self._get_info_dict_from_row_data(row_info, 'systime')
            reduce_info = self._sort_reduce_by_time(row_info_dict)
            if reduce_info:
                reduce_infos.extend(reduce_info)

        return reduce_infos

    def _get_info_dict_from_row_data(self, row_info, time_type):
        """
        Get step info in dict format.

        Args:
            row_info (list[str]): Step info, the value is corresponding to `__column__`.
            time_type (str): The value type. `systime` keeps the original value.
                `realtime` transforms the value in millisecond. Default: `realtime`.

        Returns:
            dict, step trace information. The key is in `__column__`.
        """
        row_info_dict = {}
        for key, value in zip(self.__column__, row_info):
            if key == 'step_num':
                continue
            value = to_int(value, key)
            row_info_dict[key] = to_millisecond(value) if time_type == 'realtime' else value
        return row_info_dict

    def _sort_reduce_by_time(self, row_info_dict):
        """
        Sort reduce info by time.

        Args:
            row_info_dict (dict): Step trace information.

        Returns:
            list, including the all reduce info sorted by start time only.
            [
                [reduce_field, stream_id, reduce_start, reduce_duration],
                [...],
                [...]
            ]
        """
        factor = 1e5  # convert time unit from 10ns to 1ms
        reduce_pid = 10000
        reduce_info = []
        reduce_fields = [field_name for field_name in self.__column__
                         if field_name.startswith('stream_') and not field_name.endswith('point')]
        for reduce_field in reduce_fields:
            reduce_start = row_info_dict.get(reduce_field + '_start_point')
            reduce_start = reduce_start / factor \
                if reduce_start else 0
            reduce_duration = row_info_dict.get(reduce_field)
            reduce_duration = reduce_duration / factor if reduce_duration else 0
            if not (reduce_start and reduce_duration):
                logger.info("Reduce event missing value.")
                continue
            cur_stream_id = reduce_field.split('_', 2)[1]
            reduce_meta = [reduce_field, int(cur_stream_id), reduce_start,
                           reduce_duration, reduce_pid]
            reduce_info.append(reduce_meta)

        return reduce_info

    def _query_and_sort_by_op_type(self, filter_condition, op_type_order: list):
        """
        Query the AICORE operator detail information by `filter_condition`,
        and sort by `op_type_order` and execution time.

        Args:
            filter_condition (dict): The filter condition.
            op_type_order (list[str]): The name of the operator type in order.

        Returns:
            dict, The results are filtered and sorted.
        """
        self._aicore_detail_data_load()
        if filter_condition is None:
            filter_condition = {}
        self._filter(filter_condition)

        type_detail_cache = {}
        for detail_info in self._result:
            op_type = detail_info[1]
            if op_type not in op_type_order:
                continue
            infos = type_detail_cache.get(op_type)
            if infos:
                infos.append(detail_info)
            else:
                type_detail_cache[op_type] = [detail_info]

        result = []
        for op_type in op_type_order:
            detail_infos = type_detail_cache.get(op_type)
            if detail_infos is None:
                continue
            detail_infos.sort(key=lambda item: item[2], reverse=True)
            result.extend(detail_infos)

        return {
            'col_name_detail': self._display_col_names_detail,
            'object': result
        }

    def _filter(self, filter_condition):
        """
        Filter the profiling data according to the filter condition.

        Args:
            filter_condition (dict): The filter condition.
        """
        def _inner_filter(item: list):
            return self._default_filter(item, filter_condition)

        def _inner_map(item: list):
            inner_item = item[0:4]
            if is_display_full_op_name:
                inner_item.append(item[4])
            if is_display_detail:
                inner_item.append(item[5])
            return inner_item

        is_display_detail = filter_condition.get('is_display_detail', True)
        is_display_full_op_name = filter_condition.get(
            'is_display_full_op_name', True
        )
        self._set_display_col_name(is_display_detail, is_display_full_op_name)
        if is_display_detail and is_display_full_op_name:
            self._result = list(filter(_inner_filter, self._aicore_detail_data))
        else:
            self._result = list(
                map(_inner_map, filter(_inner_filter, self._aicore_detail_data))
            )

    def _default_filter(self, item, condition):
        """
        The default filter method.

        Args:
            item (list[Union[str, float, int]]): A piece of data to be filtered.
            condition (dict): The filter condition.

        Returns:
            bool, `True` if the item is satisfied.
        """
        for condition_key, condition_value in condition.items():
            if condition_key in self._none_filter_condition_key:
                continue
            if condition_key in self._col_names_detail:
                index = self._col_names_detail.index(condition_key)
                actual_value = item[index]
                for exp_key, exp_value in condition_value.items():
                    if not self._is_match_condition(
                            exp_key, exp_value, actual_value):
                        return False
        return True

    def _is_match_condition(self, exp_key, exp_value, actual_value):
        """
        Check whether the actual value meets the expect condition.

        Args:
            exp_key (str): Expect key of the condition.
            exp_value (str): Expect value.
            actual_value (str): Actual value.

        Returns:
            bool, `True` if the actual meets the expect condition, else `False`.
        """
        if exp_key == 'in':
            if actual_value not in exp_value:
                return False
        elif exp_key == 'not_in':
            if actual_value in exp_value:
                return False
        elif exp_key == 'partial_match_str_in':
            for partial_match_str in exp_value:
                if partial_match_str in actual_value:
                    return True
            return False
        else:
            return False

        return True

    def _set_display_col_name(self, is_display_detail, is_display_full_op_name):
        """
        Set the display column name according to the filter condition.

        Args:
            is_display_detail (bool): Whether to display the detailed operator
                information.
            is_display_full_op_name (bool): Whether to display the operator full
                name.
        """
        self._display_col_names_detail = self._col_names_detail[0:4]
        if is_display_full_op_name:
            self._display_col_names_detail.append(self._col_names_detail[4])
        if is_display_detail:
            self._display_col_names_detail.append(self._col_names_detail[5])


class BaseTimelineGenerator:
    """
    Analyse timeline data from file.
    """
    __col_names__ = ['op_name', 'stream_id', 'start_time', 'duration']
    _output_timeline_data_file_path = 'output_timeline_data_{}.txt'
    _min_cycle_counter_file_path = 'min_cycle_counter_{}.txt'
    _timeline_meta = []
    _timeline_summary = {
        'total_time': 0,
        'num_of_streams': 0,
        'num_of_ops': 0,
        'op_exe_times': 0
    }

    def _load_timeline_data(self):
        """Load timeline data from file."""

    def _parse_timeline_data(self):
        """Parse timeline data."""

    def init_timeline(self):
        """Init timeline metadata, adding all collected info."""

    def write_timeline(self, size_limit=SIZE_LIMIT_DEFAULT):
        """Load data according to the parsed profiling files."""
        # Write timeline to file.
        logger.info('Writing timeline file...')
        self.write_timeline_to_json_by_limitation(size_limit)
        logger.info('Finished file writing!')

    def write_timeline_to_json_by_limitation(self, size_limit):
        """Write timeline to json by limitation."""
        display_filename = self._display_filename.format(self._device_id)
        display_file_path = os.path.join(
            self._profiling_dir,
            display_filename
        )
        display_file_path = validate_and_normalize_path(display_file_path)

        length = len(self._timeline_meta)
        try:
            with open(display_file_path, 'w') as json_file:
                json_file.write('[')
                for index, item in enumerate(self._timeline_meta):
                    json.dump(item, json_file)
                    file_size = os.path.getsize(display_file_path)
                    if file_size > size_limit:
                        break
                    if index == length - 1:
                        break
                    json_file.write(',')
                json_file.write(']')
                os.chmod(display_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.error('Error occurred when write timeline display file: %s', err)
            raise ProfilerIOException

    def write_timeline_summary(self):
        """Write timeline summary to json."""
        timeline_summary_file_path = os.path.join(
            self._profiling_dir,
            self._timeline_summary_filename.format(self._device_id)
        )

        timeline_summary_file_path = validate_and_normalize_path(timeline_summary_file_path)

        try:
            with open(timeline_summary_file_path, 'w') as json_file:
                json.dump(self._timeline_summary, json_file)
            os.chmod(timeline_summary_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.error('Error occurred when write timeline summary file: %s', err)
            raise ProfilerIOException

    @staticmethod
    def _update_num_of_streams(timeline, stream_count_dict):
        """Update number of streams."""
        stream_id = timeline[1]
        if stream_id not in stream_count_dict.keys():
            stream_count_dict[stream_id] = 1
        else:
            stream_count_dict[stream_id] += 1

    def get_min_cycle_counter(self):
        """
        Get minimum cycle counter.

        Returns:
            float, the minimum value of the cycle counter.
        """
        file_path = os.path.join(
            self._profiling_dir,
            self._min_cycle_counter_file_path.format(self._device_id)
        )

        file_path = validate_and_normalize_path(file_path)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f_obj:
                    min_cycle_counter = f_obj.read()
                    min_cycle_counter = float(min_cycle_counter) \
                        if not min_cycle_counter == 'inf' else 0
            except (IOError, OSError) as err:
                logger.error('Error occurred when read minimum cycle counter: %s', err)
                raise ProfilerIOException
        else:
            min_cycle_counter = 0
            logger.info("No min cycle counter recorded.")

        return min_cycle_counter

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
            framework_info_dict[op_full_name] = {
                'name': op_name,
                'args': {
                    'type': op_type,
                    'fullname': op_full_name
                }
            }
            framework_info_dict[op_full_name]['args'].update(op_info)

        # Insert framework info into timeline.
        for timeline_item in self._timeline_meta:
            op_full_name = timeline_item.get('name')
            framework_item = framework_info_dict.get(op_full_name)
            if framework_item:
                timeline_item['name'] = framework_item.get('name')
                timeline_item['args'] = framework_item.get('args')
        logger.debug('Finished adding framework info into timeline...')

class GpuTimelineGenerator(BaseTimelineGenerator):
    """Generate gpu Timeline data from file."""
    _display_filename = 'gpu_timeline_display_{}.json'
    _timeline_summary_filename = 'gpu_timeline_summary_{}.json'
    _output_op_execute_time_file_path = "gpu_op_execute_timestamp_{}.txt"
    _output_activity_execute_time_file_path = "activity_execute_timestamp_{}.txt"
    _output_gpu_activity_info_file_path = "gpu_activity_data_{}.csv"
    _activity_keys_list = []

    def __init__(self, profiling_dir, device_id):
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._timeline_meta = []
        self._timeline_summary = {
            'total_time': 0,
            'num_of_streams': 0,
            'num_of_ops': 0,
            'op_exe_times': 0
            }

    def _get_and_validate_path(self, file_name):
        """Generate op or activity file path from file name, and validate this path."""
        file_path = os.path.join(
            self._profiling_dir,
            file_name.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)
        if not os.path.exists(file_path):
            logger.error(f"Failed to find parsed timeline file {file_path}.")
            raise ProfilerFileNotFoundException('parsed timeline file')

        return file_path

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit of start_time(ts) from 1ns to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) / factor
        dur = op_meta.duration
        timeline_dict['dur'] = dur
        if op_meta.pid is None:
            timeline_dict['pid'] = int(self._device_id)
        else:  # AllReduce and AI CPU pid
            timeline_dict['pid'] = op_meta.pid
        if len(timeline) > 4:
            # len(timeline) > 4 refers to activity data, else op data.
            # Add args for activity data
            args_dict = {}
            for ix, value in enumerate(timeline[4:]):
                args_dict[self._activity_keys_list[ix]] = value
            timeline_dict['args'] = args_dict
        else:
            # Update total time of operator execution.
            self._timeline_summary['total_time'] += dur / factor
            self._timeline_summary['op_exe_times'] += 1

        self._timeline_meta.append(timeline_dict)

    def _load_timeline_data(self):
        """Load timeline data from file."""
        op_file_path = self._get_and_validate_path(
            self._output_op_execute_time_file_path)
        activity_file_path = self._get_and_validate_path(
            self._output_activity_execute_time_file_path)
        activity_args_file_path = self._get_and_validate_path(
            self._output_gpu_activity_info_file_path)

        timeline_list = self._load_op_data(op_file_path) + \
                self._load_activity_data(activity_file_path, activity_args_file_path)
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._device_id)
        cpu_timeline_list = cpu_timeline_generator.load_cpu_op_data()
        if cpu_timeline_list:
            self._clock_synchronize_to_gpu(cpu_timeline_list)
            timeline_list.extend(cpu_timeline_list)
        timeline_list.sort(key=lambda x: float(x[2]))

        return timeline_list

    def _clock_synchronize_to_gpu(self, timeline_list):
        """Synchronize the timestamp from device to host."""
        start_time_file_path = os.path.join(self._profiling_dir, f"start_time_{self._device_id}.txt")

        try:
            with open(start_time_file_path) as f:
                lines = f.readlines()
                host_monotonic_start_time = int(lines[0].strip().split(':')[-1])
                gpu_start_time = int(lines[1].strip().split(':')[-1])
        except (IOError, OSError) as err:
            logger.error(f'Error occurred when read {start_time_file_path}: {err}')
            raise ProfilerIOException

        time_diff = gpu_start_time - host_monotonic_start_time
        start_time = 2
        for idx, time_item in enumerate(timeline_list):
            timeline_list[idx][start_time] = int(time_item[start_time]) + time_diff

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
                        line_list = op_list[:2] + time
                        op_timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.error('Error occurred when load operator timeline data intermediate file: %s', err)
            raise ProfilerIOException

        return op_timeline_list

    def _load_activity_data(self, activity_file_path, activity_args_file_path):
        """Load activity data from file"""
        activity_timeline_list = []
        try:
            args_dict = {}
            with open(activity_args_file_path, 'r') as args_file:
                csv_reader = csv.reader(args_file)
                keys_list = next(csv_reader)
                # keys_list [name, type, op_full_name, stream_id, block_dim, grid_dim, ...]
                self._activity_keys_list = keys_list[1:3] + keys_list[4:6]
                for info in csv_reader:
                    args_dict[info[0]] = info[1:3] + info[4:6]
            with open(activity_file_path, 'r') as f_obj:
                for line in f_obj:
                    line_list = line.strip('\n').split(';')
                    # concat activity args info.
                    line_list += args_dict[line_list[0]]
                    activity_timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.error('Error occurred when load activity timeline data intermediate file: %s', err)
            raise ProfilerIOException

        return activity_timeline_list


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

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict.keys())

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

class AscendTimelineGenerator(BaseTimelineGenerator):
    """Generate ascend Timeline data from file."""
    _display_filename = 'ascend_timeline_display_{}.json'
    _timeline_summary_filename = 'ascend_timeline_summary_{}.json'

    def __init__(self, profiling_dir, device_id):
        self._profiling_dir = profiling_dir
        self._device_id = device_id

    def _load_timeline_data(self):
        """Load timeline data from file."""
        file_path = os.path.join(
            self._profiling_dir,
            self._output_timeline_data_file_path.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)
        if not os.path.exists(file_path):
            logger.error("Failed to find parsed timeline file.")
            raise ProfilerFileNotFoundException('parsed timeline file')

        timeline_list = []
        try:
            with open(file_path, 'r') as f_obj:
                for line in f_obj:
                    if not line.startswith('op_name'):
                        line_list = line.strip('\n').split(',')
                        timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.error('Error occurred when read timeline intermediate file: %s', err)
            raise ProfilerIOException

        return timeline_list

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit from 1ms to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) * factor
        dur = op_meta.duration * factor
        timeline_dict['dur'] = dur
        if op_meta.pid is None:
            timeline_dict['pid'] = int(self._device_id)
            # Update total time of operator execution.
            self._timeline_summary['total_time'] += dur
        else:  # AllReduce and AI CPU pid
            timeline_dict['pid'] = op_meta.pid
        self._timeline_meta.append(timeline_dict)

    def init_timeline(self, all_reduce_info, framework_info, aicpu_info, min_cycle_counter, source_path):
        """
        Init timeline metadata, adding all collected info.

        Args:
            all_reduce_info (list[list]): The metadata of AllReduce operator.
            framework_info (dict): The framework metadata.
            aicpu_info (dict): The metadata of AI CPU operator.
            min_cycle_counter (float): The minimum cycle counter of the timeline.
        """
        if min_cycle_counter == float('inf'):
            min_cycle_counter = 0

        logger.info('Initiating timeline...')
        timeline_list = self._load_timeline_data()
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._device_id)
        cpu_timeline_list = cpu_timeline_generator.get_timeline_data()
        if cpu_timeline_list:
            self._clock_synchronize_to_host(timeline_list, source_path)
            timeline_list.extend(cpu_timeline_list)
        timeline_list.sort(key=lambda x: float(x[2]))
        self._timeline_summary['op_exe_times'] = len(timeline_list)

        # Add AllReduce info to timeline temp list and sort by start time.
        if all_reduce_info:
            logger.debug('AllReduce info found. Start adding info into timeline...')
            timeline_list.extend(all_reduce_info)
            timeline_list.sort(key=lambda x: float(x[2]))

        # Add AI CPU data into timeline temp list and sort by start time.
        aicpu_data = aicpu_info.get('info')
        if aicpu_data:
            timeline_list.extend(aicpu_data)
            timeline_list.sort(key=lambda x: float(x[2]))
            self._timeline_summary['op_exe_times'] += aicpu_info.get('op_exe_times', 0)
            self._timeline_summary['num_of_streams'] += aicpu_info.get('num_of_streams', 0)
            self._timeline_summary['num_of_ops'] += aicpu_info.get('num_of_ops', 0)
            self._timeline_summary['total_time'] += aicpu_info.get('total_time', 0)

        # Init a dict for counting the num of streams.
        stream_count_dict = {}
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, min_cycle_counter)
            # Updating the collection of streams.
            if len(timeline) == 4:
                self._update_num_of_streams(timeline, stream_count_dict)

        # Get framework metadata.
        framework_obj_list = framework_info.get('object')
        # The length of list is the number of operators.
        self._timeline_summary['num_of_ops'] += len(framework_obj_list)
        self._add_framework_info(framework_obj_list)
        logger.info('Finished adding info into timeline...')

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict.keys())

    def _clock_synchronize_to_host(self, timeline_list, source_path):
        """Synchronize the timestamp from device to host."""
        host_start_file_path = os.path.join(source_path, f"host_start.log.{self._device_id}")
        dev_start_file_path = os.path.join(source_path, f"dev_start.log.{self._device_id}")

        try:
            with open(host_start_file_path) as f:
                lines = f.readlines()
                host_monotonic = int(lines[2].strip().split(':')[1])
        except (IOError, OSError) as err:
            logger.error('Error occurred when read host_start.log: %s', err)
            raise ProfilerIOException
        try:
            with open(dev_start_file_path) as f:
                lines = f.readlines()
                dev_cntvct = int(lines[2].strip().split(':')[1])
        except (IOError, OSError) as err:
            logger.error('Error occurred when read dev_start.log: %s', err)
            raise ProfilerIOException

        factor_ns_to_ms = 1e6
        factor_ms_to_ten_ns = 1e5
        factor_ten_ns_to_ns = 10
        start_time = 2
        for idx, time_item in enumerate(timeline_list):
            cycle_counter = int(float(time_item[start_time]) * factor_ms_to_ten_ns)
            host_monotonic_time = host_monotonic + (cycle_counter - dev_cntvct) * factor_ten_ns_to_ns
            timeline_list[idx][start_time] = host_monotonic_time / factor_ns_to_ms

class CpuTimelineGenerator(GpuTimelineGenerator):
    """Generate gpu Timeline data from file."""
    _output_op_execute_time_file_path = "cpu_op_execute_timestamp_{}.txt"

    def _get_and_validate_path(self, file_name):
        """Generate op or activity file path from file name, and validate this path."""
        file_path = os.path.join(
            self._profiling_dir,
            file_name.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)

        return file_path

    def load_cpu_op_data(self):
        """Load cpu operator data from file"""
        op_file_path = self._get_and_validate_path(
            self._output_op_execute_time_file_path)
        timeline_list = []
        if not os.path.exists(op_file_path):
            logger.info("No cpu operator info.")
            return timeline_list
        timeline_list = self._load_op_data(op_file_path)

        return timeline_list

    def get_timeline_data(self):
        """Get timeline data from file."""
        timeline_list = self.load_cpu_op_data()
        factor_ns_to_ms = 1e6
        start_time = 2
        duration = 3
        for idx, time_item in enumerate(timeline_list):
            time_item[start_time] = float(time_item[start_time]) / factor_ns_to_ms
            time_item[duration] = float(time_item[duration])
            timeline_list[idx] = time_item

        return timeline_list
