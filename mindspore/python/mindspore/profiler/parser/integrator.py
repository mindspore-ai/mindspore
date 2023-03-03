# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from decimal import Decimal
from enum import Enum
import sys
from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerRawFileException
from mindspore.profiler.common.util import query_latest_trace_time_file, to_int, to_millisecond
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


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
    _header_aicore_type = ['op_type', 'total_time', 'execution_frequency', 'percent']
    _header_aicore_detail = ['full_op_name', 'execution_time', 'execution_frequency']
    _header_aicpu = ['serial_number', 'op_type', 'total_time', 'dispatch_time',
                     'execution_time', 'run_start', 'run_end']

    _file_name_aicore_type_time = 'aicore_intermediate_{}_type.csv'
    _file_name_aicore_detail_info = 'aicore_intermediate_{}_detail.csv'
    _col_names_detail = ['op_name', 'op_type', 'avg_execution_time', 'subgraph', 'full_op_name', 'op_info']
    _none_filter_condition_key = ['is_display_detail', 'is_display_full_op_name']
    _none_sort_col_names = ['op_info']
    _aicore_data = []
    _aicore_detail_data = []
    _aicore_trace_data = []

    def __init__(self, profiling_dir, device_id):
        csv.field_size_limit(sys.maxsize)
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._op_time_cache = {}
        self._total_time = Decimal('0.0')
        self._column = ""
        self._result = []

    @staticmethod
    def _is_match_condition(exp_key, exp_value, actual_value):
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

    def integrate(self):
        """Integrate the parsed profiling files."""
        self._parse_aicore_detail_time()
        self._parse_aicore_type_time()
        self._parse_aicpu_time()

    def get_aicore_data(self):
        """Get ai core data."""
        self._aicore_data_load()
        return self._aicore_data

    def get_aicore_detail_data(self):
        """Get ai core detail data."""
        self._aicore_detail_data_load()
        return self._aicore_detail_data

    def get_aicore_trace_data(self):
        """Get ai core trace data."""
        self._aicore_trace_data_load()
        return self._aicore_trace_data

    def query_for_all_reduce(self):
        """Query all reduce data."""
        return self._query_for_all_reduce()

    def query_and_sort_by_op_type(self, filter_condition, op_type_order):
        """Query and sort by op type."""
        return self._query_and_sort_by_op_type(filter_condition, op_type_order)

    def _parse_aicore_type_time(self):
        """Parse the parsed AICORE operator type file."""
        if self._total_time == 0:
            logger.info("AICORE data does not exist.")
            return

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
        for full_op_name, op_info in self._op_time_cache.items():
            self._total_time += op_info[0] * op_info[1]
            op_type = op_name_type_cache.get(full_op_name)
            op_type_time = op_type_time_cache.get(op_type)
            if not op_type_time:
                op_type_time = [op_info[0] * op_info[1], op_info[1]]
                op_type_time_cache[op_type] = op_type_time
            else:
                op_type_time[0] += op_info[0] * op_info[1]
                op_type_time[1] += op_info[1]
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
                    self._op_time_cache[op_infos[0]] = [Decimal(op_infos[1]), int(op_infos[3])]
                    csv_writer.writerow([op_infos[0], op_infos[1], op_infos[3]])

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
            logger.warning("Failed to find parsed trace time file.")
            return
        file_path = validate_and_normalize_path(file_path)
        with open(file_path, 'r') as handle:
            csv_reader = csv.reader(handle)
            self._column = next(csv_reader)
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
                    raise ProfilerRawFileException('Fail to parse point info file.') from err

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
        reduce_infos.sort(key=lambda x: float(x[2]))
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
        for key, value in zip(self._column, row_info):
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
        reduce_fields = [field_name for field_name in self._column
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
            cur_stream_id = reduce_field.split('_', 3)[1]
            if reduce_field.split('_', 2)[1] == 'ops':
                cur_stream_id = reduce_field.split('_', 3)[2]
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


class DeviceTarget(Enum):
    """The device target enum."""
    CPU = 'cpu'
    GPU = 'gpu'
    ASCEND = 'ascend'
