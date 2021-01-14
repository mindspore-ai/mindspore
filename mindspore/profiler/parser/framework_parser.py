# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Thr parser for parsing framework files."""
import csv
import enum
import json
import os
import re
import stat

from mindspore.profiler.common.exceptions.exceptions import \
    ProfilerPathErrorException, ProfilerDirNotFoundException, \
    ProfilerFileNotFoundException, ProfilerDeviceIdMismatchException, \
    ProfilerRawFileException, ProfilerParamValueErrorException
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path


class VmDataType(enum.IntEnum):
    """Definition of vm data type."""
    NUMBER_TYPE_BEGIN = 26
    NUMBER_TYPE_BOOL = 27
    NUMBER_TYPE_INT = 28
    NUMBER_TYPE_INT8 = 29
    NUMBER_TYPE_INT16 = 30
    NUMBER_TYPE_INT32 = 31
    NUMBER_TYPE_INT64 = 32
    NUMBER_TYPE_UINT = 33
    NUMBER_TYPE_UINT8 = 34
    NUMBER_TYPE_UINT16 = 35
    NUMBER_TYPE_UINT32 = 36
    NUMBER_TYPE_UINT64 = 37
    NUMBER_TYPE_FLOAT = 38
    NUMBER_TYPE_FLOAT16 = 39
    NUMBER_TYPE_FLOAT32 = 40
    NUMBER_TYPE_FLOAT64 = 41
    NUMBER_TYPE_END = 42

    @classmethod
    def get_data_type_name(cls, num):
        """
        Get the name of data type by enum number.

        Args:
            num (int): Enum number.

        Returns:
            str, the name of data type.
        """
        data_type = cls._value2member_map_.get(num)
        return 'UNKNOWN' if data_type is None else data_type.name


class GeDataType(enum.IntEnum):
    """Definition of ge data type."""
    DT_FLOAT = 0
    DT_FLOAT16 = 1
    DT_INT8 = 2
    DT_INT16 = 6
    DT_UINT16 = 7
    DT_UINT8 = 4
    DT_INT32 = 3
    DT_INT64 = 9
    DT_UINT32 = 8
    DT_UINT64 = 10
    DT_BOOL = 12
    DT_DOUBLE = 11
    DT_STRING = 13
    DT_DUAL_SUB_INT8 = 14
    DT_DUAL_SUB_UINT8 = 15
    DT_COMPLEX64 = 16
    DT_COMPLEX128 = 17
    DT_QINT8 = 18
    DT_QINT16 = 19
    DT_QINT32 = 20
    DT_QUINT8 = 21
    DT_QUINT16 = 22
    DT_RESOURCE = 23
    DT_STRING_REF = 24
    DT_DUAL = 25
    DT_UNDEFINED = 26

    @classmethod
    def get_data_type_name(cls, num):
        """
        Get the name of data type by enum number.

        Args:
            num (int): Enum number.

        Returns:
            str, the name of data type.
        """
        data_type = cls._value2member_map_.get(num)
        return 'UNKNOWN' if data_type is None else data_type.name


class GeFormat(enum.IntEnum):
    """Definition of ge format type."""
    FORMAT_NCHW = 0
    FORMAT_NHWC = 1
    FORMAT_ND = 2
    FORMAT_NC1HWC0 = 3
    FORMAT_FRACTAL_Z = 4
    FORMAT_NC1C0HWPAD = 5
    FORMAT_NHWC1C0 = 6
    FORMAT_FSR_NCHW = 7
    FORMAT_FRACTAL_DECONV = 8
    FORMAT_C1HWNC0 = 9
    FORMAT_FRACTAL_DECONV_TRANSPOSE = 10
    FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11
    FORMAT_NC1HWC0_C04 = 12
    FORMAT_FRACTAL_Z_C04 = 13
    FORMAT_CHWN = 14
    FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15
    FORMAT_HWCN = 16
    FORMAT_NC1KHKWHWC0 = 17
    FORMAT_BN_WEIGHT = 18
    FORMAT_FILTER_HWCK = 19
    FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20
    FORMAT_HASHTABLE_LOOKUP_KEYS = 21
    FORMAT_HASHTABLE_LOOKUP_VALUE = 22
    FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23
    FORMAT_HASHTABLE_LOOKUP_HITS = 24
    FORMAT_C1HWNCOC0 = 25
    FORMAT_MD = 26
    FORMAT_NDHWC = 27
    FORMAT_FRACTAL_ZZ = 28
    FORMAT_FRACTAL_NZ = 29
    FORMAT_NCDHW = 30
    FORMAT_DHWCN = 31
    FORMAT_NDC1HWC0 = 32
    FORMAT_FRACTAL_Z_3D = 33
    FORMAT_CN = 34
    FORMAT_NC = 35
    FORMAT_DHWNC = 36
    FORMAT_FRACTAL_Z_3D_TRANSPOSE = 37
    FORMAT_RESERVED = 38
    FORMAT_ALL = 39

    @classmethod
    def get_format_name(cls, num):
        """
        Get the name of format type by enum number.

        Args:
            num (int): Enum number.

        Returns:
            str, the name of format type.
        """
        format_type = cls._value2member_map_.get(num)
        return 'UNKNOWN' if format_type is None else format_type.name


class FrameworkParser:
    """
    Thr parser for parsing framework files.

    Args:
        profiling_id (str): The profiling ID.
        device_id (str): The device ID.
        output_path (str): The directory of the parsed file. Default: `./`.
    """
    _regex_framework = r'Framework\.(?P<data_type>.+)\.(?P<device_id>\d).+'
    _regex_framework_in_data = r'Framework\.(?P<data_type>.+)\.' \
                               r'(?P<device_id>\d)\.(?P<profiling_id>[a-zA-Z0-9]+).+'
    _col_names = [
        'task_id', 'stream_id', 'block_dim', 'full_op_name', 'op_name',
        'op_type', 'subgraph', 'op_info'
    ]
    _graph_attr_name = [
        'input_format', 'input_data_type', 'input_shape', 'output_format',
        'output_data_type', 'output_shape'
    ]

    # if the task id is less than the task id threshold, The combination of
    # task id and Stream id represents one operator, else the task id represents
    # one operator
    _task_id_threshold = 25000

    def __init__(self, profiling_id, device_id, output_path='./'):
        self._raw_data_dir = output_path
        self._profiling_path = self._get_raw_profiling_path(profiling_id)
        self._backend_type = None
        self._framework_path = {'graph': [], 'task': [], 'point': []}
        self._search_file(profiling_id, device_id)
        self._device_id = device_id
        self._save_path = self._get_save_path(device_id, output_path)
        self._task_id_full_op_name_dict = {}
        self._task_cache = {}
        self._point_info = {}
        self._parse_task_files()
        self._parse_point_files()

    @property
    def save_path(self):
        """
        The property of save path.

        Returns:
            str, the save path.
        """
        return self._save_path

    @property
    def point_info(self):
        """
        The property of the framework point information.

        Returns:
            dict, the framework point information.
        """
        return self._point_info

    def to_task_id_full_op_name_dict(self):
        """
        Get the task id and full operator name dict.

        Returns:
            dict, the task id and full operator name dict.
        """
        return self._task_id_full_op_name_dict

    def parse(self):
        """Parse the framework files."""
        self._parse_graph_files_and_save(self._task_cache)
        del self._task_cache

    def check_op_name(self, op_name, is_prefix=True):
        """
        Check whether the operator name exists.

        Args:
            op_name (str): The operator name or operator name prefix.
            is_prefix (bool): `True` if the op_name is prefix, else `False`.
                Default: True.

        Returns:
            bool, `True` if the operator name does exist in framework file, else
            `False`.
        """
        if not op_name:
            raise ProfilerParamValueErrorException('The op_name should exist.')
        for full_op_name in self._task_id_full_op_name_dict.values():
            if full_op_name:
                if is_prefix and full_op_name.startswith(op_name):
                    return True
                if not is_prefix and op_name == full_op_name:
                    return True
        return False

    def _get_raw_profiling_path(self, profiling_id):
        """
        Get raw profiling path.

        Args:
            profiling_id (str): The profiling ID.

        Returns:
            str, the raw profiling path.

        Raises:
            ProfilerPathErrorException: If the profiling path is invalid.
            ProfilerDirNotFoundException: If the profiling dir is not found.
        """
        profiling_path = os.path.join(self._raw_data_dir, profiling_id)
        try:
            profiling_path = validate_and_normalize_path(profiling_path)
        except RuntimeError:
            raise ProfilerPathErrorException('Profiling path is invalid.')
        if not os.path.isdir(profiling_path):
            raise ProfilerDirNotFoundException(profiling_path)
        return profiling_path

    def _search_file(self, profiling_id, device_id):
        """
        Search all framework files in raw profiling path.

        Args:
            profiling_id (str): The profiling ID.
            device_id (str): The device ID.

        Raises:
            ProfilerFileNotFoundException: If the framework files are not found.
        """
        # first search in the JOB dir, and if not, search in the sub directory
        # in the JOB
        self._search_file_from_job_path(device_id, search_in_sub_path=False)
        if self._backend_type is None:
            self._search_file_from_job_path(device_id, search_in_sub_path=True)
        self._search_file_from_data_path(profiling_id, device_id)

        if self._backend_type is None:
            raise ProfilerFileNotFoundException('Framework')
        self._framework_path['graph'].sort()
        self._framework_path['task'].sort()

    def _search_file_from_job_path(self, device_id, search_in_sub_path=False):
        """
        Search framework files from job path.

        Args:
            device_id (str): The device ID.
            search_in_sub_path (bool): `True` if search file in profiling dir,
                else search in profiling sub dir. Default: False.

        Raises:
            ProfilerRawFileException: If the framework file type is inconsistent.
            ProfilerDeviceIdMismatchException: If the device id is mismatch
                with framework in the raw dir.
        """
        profiling_dir = os.path.join(self._profiling_path, 'data') \
            if search_in_sub_path else self._profiling_path
        if not os.path.isdir(profiling_dir):
            return

        files = os.listdir(profiling_dir)
        for file in files:
            pattern = re.search(self._regex_framework, file)
            if not pattern or file.endswith('.done'):
                continue
            attrs = pattern.groupdict()

            device_id_in_path = attrs.get('device_id')
            if device_id_in_path != device_id:
                raise ProfilerDeviceIdMismatchException()

            data_type = attrs.get('data_type')
            data_type = data_type.replace("host.", "")
            if data_type.startswith('vm_'):
                if self._backend_type and self._backend_type != 'vm':
                    raise ProfilerRawFileException('Backend type is inconsistent.')
                self._backend_type = 'vm'
                _, data_type = data_type.split('_', 1)
            else:
                if self._backend_type and self._backend_type != 'ge':
                    raise ProfilerRawFileException('Backend type is inconsistent.')
                self._backend_type = 'ge'
            if data_type.startswith('graph_desc_info'):
                self._framework_path['graph'].append(
                    os.path.join(profiling_dir, file)
                )
            elif data_type.startswith('task_desc_info'):
                self._framework_path['task'].append(
                    os.path.join(profiling_dir, file)
                )
            elif data_type.startswith('point'):
                self._framework_path['point'].append(
                    os.path.join(profiling_dir, file)
                )

    def _search_file_from_data_path(self, profiling_id, device_id):
        """
        Search framework files from data path.

        Args:
            profiling_id (str): The profiling ID.
            device_id (str): The device ID.

        Raises:
            ProfilerRawFileException: If the framework file type is inconsistent.
            ProfilerDeviceIdMismatchException: If the device id is mismatch
                with framework in the raw dir.
        """
        profiling_data_path = os.path.join(
            self._raw_data_dir, 'container', device_id, 'data'
        )
        if not os.path.isdir(profiling_data_path):
            return

        files = os.listdir(profiling_data_path)
        for file in files:
            pattern = re.search(self._regex_framework_in_data, file)
            if not pattern or file.endswith('.done') or file.endswith('.zip'):
                continue
            attrs = pattern.groupdict()

            profiling_id_in_path = attrs.get('profiling_id')
            if profiling_id_in_path != profiling_id:
                continue

            device_id_in_path = attrs.get('device_id')
            if device_id_in_path != device_id:
                raise ProfilerDeviceIdMismatchException()

            data_type = attrs.get('data_type')
            data_type = data_type.replace("host.", "")
            if data_type.startswith('vm_'):
                if self._backend_type and self._backend_type != 'vm':
                    raise ProfilerRawFileException('Backend type is inconsistent.')
                self._backend_type = 'vm'
                _, data_type = data_type.split('_', 1)
            else:
                if self._backend_type and self._backend_type != 'ge':
                    raise ProfilerRawFileException('Backend type is inconsistent.')
                self._backend_type = 'ge'
            if data_type.startswith('graph_desc_info'):
                self._framework_path['graph'].append(
                    os.path.join(profiling_data_path, file)
                )
            elif data_type.startswith('task_desc_info'):
                self._framework_path['task'].append(
                    os.path.join(profiling_data_path, file)
                )
            elif data_type.startswith('point'):
                self._framework_path['point'].append(
                    os.path.join(profiling_data_path, file)
                )

    def _get_save_path(self, device_id, output_path):
        """
        Get the save path.

        Args:
            device_id (str): The device ID.
            output_path (str): The output dir.

        Returns:
            str, the save path.

        Raises:
            ProfilerPathErrorException: If the output path is invalid.
            ProfilerDirNotFoundException: If the output dir is not found.
        """
        try:
            output_dir = validate_and_normalize_path(output_path)
        except RuntimeError:
            raise ProfilerPathErrorException('Output path is invalid.')
        if not os.path.isdir(output_dir):
            raise ProfilerDirNotFoundException(output_dir)
        return os.path.join(
            output_dir, '_'.join(['framework', 'raw', device_id]) + '.csv'
        )

    def _parse_task_files(self):
        """Parse the framework task files."""
        for path in self._framework_path['task']:
            path = validate_and_normalize_path(path)
            with open(path, 'r') as file:
                for task_info in file:
                    infos = task_info.strip('\n').split(' ')
                    infos = infos[1:] if len(infos) == 5 else infos
                    # key is op name, values is task id, stream id, block_dim
                    self._task_cache[infos[0]] = [infos[2], infos[3], infos[1]]

                    # if the task id is less than the task id threshold, the
                    # stream id and task id correspond to an operator
                    task_id = infos[2]
                    if int(task_id) < self._task_id_threshold:
                        task_id = '_'.join([infos[3], task_id])
                    self._task_id_full_op_name_dict[task_id] = infos[0]

    def _parse_graph_files_and_save(self, task_cache):
        """
        Parse the framework graph files and save the framework information.

        Args:
            task_cache (dict): The task information cache.
        """
        with open(self._save_path, 'w') as save_file:
            csv_writer = csv.writer(save_file)
            csv_writer.writerow(self._col_names)
            pre_graph_info = None
            for path in self._framework_path['graph']:
                first_row = True
                with open(path, 'r') as graph_file:
                    for graph_info in graph_file:
                        if first_row is True:
                            first_row = False
                            # The last row of the previous file and the first row of the current file may need
                            # to be combined to one row
                            if graph_info.startswith("op_name:") is False:
                                pre_graph_info = pre_graph_info + graph_info
                                continue
                        if pre_graph_info is not None:
                            self._parse_graph_row_and_save(task_cache, csv_writer, pre_graph_info)
                        pre_graph_info = graph_info

            if pre_graph_info is not None:
                self._parse_graph_row_and_save(task_cache, csv_writer, pre_graph_info)

            none_list = [None, None, None, None]
            for key, value in task_cache.items():
                value.append(key)
                value.extend(none_list)
                csv_writer.writerow(value)
        os.chmod(self._save_path, stat.S_IREAD | stat.S_IWRITE)

    def _parse_graph_row_and_save(self, task_cache, csv_writer, graph_info):
        """
        Parse the framework graph row and save the framework information.

        Args:
            task_cache (dict): The task information cache.
            csv_writer (csv): Csv writer.
            graph_info (str): Row info of graph.
        """
        result = self._parse_one_row_graph_info(graph_info)
        task_info = task_cache.get(result[0])
        if task_info:
            task_info.extend(result)
            csv_writer.writerow(task_info)
            del task_cache[result[0]]
        else:
            save_info = [None, None, None]
            save_info.extend(result)
            csv_writer.writerow(save_info)

    def _parse_one_row_graph_info(self, row_info):
        """
        Parse the graph information in one row.

        Args:
            row_info (str): One row graph information.

        Returns:
            list[str], the parsed graph information.
        """
        full_op_name = None
        op_name = None
        subgraph_name = None
        op_type = None
        op_info = dict()
        cur_op_info_key = None

        infos = row_info.strip('\n').split(' ')
        for info in infos:
            attr_name, attr_value = info.split(':', 1)
            if attr_name == 'op_name':
                full_op_name = attr_value
                subgraph_name = self._get_subgraph_name(full_op_name)
                op_name = self._get_op_name(full_op_name, subgraph_name)
            elif attr_name == 'op_type':
                op_type = attr_value
            elif attr_name in ['input_id', 'output_id']:
                cur_op_info_key = '{}_{}'.format(
                    attr_name.split('_')[0], attr_value
                )
                op_info[cur_op_info_key] = dict()
            elif attr_name in self._graph_attr_name:
                op_attr = attr_name.split('_', 1)[1]
                if op_attr == 'shape':
                    attr_value = attr_value.strip('"')
                if self._backend_type == 'vm':
                    if op_attr == 'data_type':
                        attr_value = VmDataType.get_data_type_name(
                            int(attr_value)
                        )
                else:
                    if op_attr == 'data_type':
                        attr_value = GeDataType.get_data_type_name(
                            int(attr_value)
                        )
                    elif op_attr == 'format':
                        attr_value = GeFormat.get_format_name(int(attr_value))

                op_info[cur_op_info_key][op_attr] = attr_value

        # the list info are full_op_name, op_name, op_type, subgraph, op_info
        return [full_op_name, op_name, op_type, subgraph_name,
                json.dumps(op_info)]

    def _get_subgraph_name(self, full_op_name):
        """
        Get subgraph name.

        Args:
            full_op_name (str): The full operator name.

        Returns:
            str, the subgraph name.
        """
        subgraph_name = full_op_name.split('/', 1)[0]
        if subgraph_name in ['Default', 'Gradients']:
            return subgraph_name
        return None

    def _get_op_name(self, full_op_name, subgraph_name):
        """
        Get operator name.

        Args:
            full_op_name (str): The full operator name.
            subgraph_name (str): The subgraph name.

        Returns:
            str, the operator name.
        """
        if subgraph_name is None:
            return full_op_name

        if self._backend_type == 'vm':
            return full_op_name.split('/')[-1]

        strs = full_op_name.split(subgraph_name + '/')
        op_name = None
        for name_str in strs:
            if not name_str:
                continue
            if op_name is None:
                op_name = name_str.split('/')[-1]
            else:
                op_name = '+'.join([op_name, name_str.split('/')[-1]])
        return op_name

    def _parse_point_files(self):
        """Parse the framework point files."""
        for path in self._framework_path['point']:
            path = validate_and_normalize_path(path)
            with open(path, 'r') as file:
                for point_info in file:
                    infos = point_info.strip('\n').split(' ')
                    self._point_info[int(infos[0])] = infos[1]
