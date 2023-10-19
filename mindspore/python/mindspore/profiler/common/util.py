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
"""
Profiler util.

This module provides the utils.
"""
import os

# one sys count takes 10 ns, 1 ms has 100000 system count
import re
import stat

from mindspore import log as logger


def to_int(param, param_name):
    """
    Transfer param to int type.

    Args:
        param (Any): A param transformed.
        param_name (str): Param name.

    Returns:
        int, value after transformed.

    """
    try:
        param = int(param)
    except ValueError as err:
        raise TypeError('Must be Integer: ' + param_name) from err
    return param


def fwrite_format(output_data_path, data_source=None, is_print=False, is_start=False):
    """
    Write data to the output file.

    Args:
         output_data_path (str): The output file path of the data.
         data_source (str, list, tuple): The data to write.
         is_print (bool): whether to print the data to stdout.
         is_start (bool): Whether is the first line of the output file, will remove the old file if True."
    """

    if is_start and os.path.exists(output_data_path):
        os.remove(output_data_path)

    if isinstance(data_source, str) and data_source.startswith("title:"):
        title_label = '=' * 20
        data_source = title_label + data_source[6:] + title_label

    with open(output_data_path, 'a+') as f:
        if isinstance(data_source, (list, tuple)):
            for raw_data in data_source:
                if isinstance(raw_data, (list, tuple)):
                    raw_data = map(str, raw_data)
                    raw_data = " ".join(raw_data)
                f.write(raw_data)
                f.write("\n")
        else:
            f.write(data_source)
            f.write("\n")
    os.chmod(output_data_path, stat.S_IREAD | stat.S_IWRITE)

    if is_print:
        if isinstance(data_source, (list, tuple)):
            for raw_data in data_source:
                if isinstance(raw_data, (list, tuple)):
                    raw_data = map(str, raw_data)
                    raw_data = " ".join(raw_data)
                logger.info(raw_data)
        else:
            logger.info(data_source)


def get_log_slice_id(file_name):
    """Get log slice id."""
    pattern = re.compile(r'(?<=slice_)\d+')
    slice_list = pattern.findall(file_name)
    index = re.findall(r'\d+', slice_list[0])
    return int(index[0])


def get_file_join_name(input_path, file_name):
    """
    Search files under the special path, and will join all the files to one file.

    Args:
        input_path (str): The source path, will search files under it.
        file_name (str): The target of the filename, such as 'hwts.log.data.45.dev'.

    Returns:
        str, the join file name.
    """
    name_list = []
    file_join_name = ''
    input_path = os.path.realpath(input_path)
    if os.path.exists(input_path):
        files = os.listdir(input_path)
        for f in files:
            if file_name in f and not f.endswith('.done') and not f.endswith('.join') \
                    and not f.endswith('.zip'):
                name_list.append(f)

        # resort name_list
        name_list.sort(key=get_log_slice_id)

    if len(name_list) == 1:
        file_join_name = os.path.join(input_path, name_list[0])
    elif len(name_list) > 1:
        file_join_name = os.path.join(input_path, '%s.join' % file_name)
        if os.path.exists(file_join_name):
            os.remove(file_join_name)
        file_join_name = os.path.realpath(file_join_name)
        with open(file_join_name, 'ab') as bin_data:
            for i in name_list:
                file = input_path + os.sep + i
                with open(file, 'rb') as txt:
                    bin_data.write(txt.read())
    return file_join_name


def get_file_path(input_path, file_name):
    """
    Search files under the special path.

    Args:
        input_path (str): The source path, will search files under it.
        file_name (str): The target of the filename, such as 'host_start_log'.

    Returns:
        str, a special file path. If there can not find the special path, will return None.
    """

    input_path = os.path.realpath(input_path)
    if os.path.exists(input_path):
        files = os.listdir(input_path)
        for f in files:
            if file_name in f and not f.endswith('.done') \
                    and not f.endswith('.zip'):
                return os.path.join(input_path, f)

    return None


def parse_device_id(filename, device_id_list, profiler_file_prefix):
    """Parse device id from filename."""
    items = filename.split("_")
    if filename.startswith("step_trace_raw"):
        device_num = ""
        if len(items) > 3:
            device_num = items[3]
    else:
        device_num = items[-1].split(".")[0] if items[-1].split(".") else ""

    if device_num.isdigit() and '_'.join(items[:-1]) in profiler_file_prefix:
        device_id_list.add(device_num)


def analyse_device_list_from_profiler_dir(profiler_dir):
    """
    Analyse device list from profiler dir.

    Args:
        profiler_dir (str): The profiler data dir.

    Returns:
        list, the device_id list.
    """
    profiler_file_prefix = ["timeline_display", "output_op_compute_time"]

    device_id_list = set()
    for _, _, filenames in os.walk(profiler_dir):
        for filename in filenames:
            parse_device_id(filename, device_id_list, profiler_file_prefix)

    return sorted(list(device_id_list))


def query_latest_trace_time_file(profiler_dir, device_id=0):
    """
    Query the latest trace time file.

    Args:
        profiler_dir (str): The profiler directory.
        device_id (int): The id of device.

    Returns:
        str, the latest trace time file path.
    """
    files = os.listdir(profiler_dir)
    target_file = f'step_trace_raw_{device_id}_detail_time.csv'
    try:
        latest_file = max(
            filter(
                lambda file: file == target_file,
                files
            ),
            key=lambda file: os.stat(os.path.join(profiler_dir, file)).st_mtime
        )
    except ValueError:
        return None
    return os.path.join(profiler_dir, latest_file)


def query_step_trace_file(profiler_dir):
    """
    Query for all step trace file.

    Args:
        profiler_dir (str): The directory that contains all step trace files.

    Returns:
        str, the file path of step trace time.
    """
    files = os.listdir(profiler_dir)
    training_trace_file = list(
        filter(
            lambda file: file.startswith('training_trace') and not file.endswith('.done'),
            files
        )
    )
    if training_trace_file:
        return os.path.join(profiler_dir, training_trace_file[0])
    return None


def get_summary_for_step_trace(average_info, header, is_training_mode=True):
    """The property of summary info."""
    if not average_info or not header:
        return {}
    total_time = get_field_value(average_info, 'total', header)
    iteration_interval = get_field_value(average_info, 'iteration_interval',
                                         header)
    summary_part = {
        'total_time': total_time,
        'iteration_interval': iteration_interval,
        'iteration_interval_percent': calculate_percent(iteration_interval, total_time),
    }
    if is_training_mode:
        fp_and_bp = get_field_value(average_info, 'fp_and_bp', header)
        tail = get_field_value(average_info, 'tail', header)
        summary = {
            'fp_and_bp': fp_and_bp,
            'fp_and_bp_percent': calculate_percent(fp_and_bp, total_time),
            'tail': tail,
            'tail_percent': calculate_percent(tail, total_time)
        }
    else:
        fp = get_field_value(average_info, 'fp', header)
        summary = {
            'fp': fp,
            'fp_percent': calculate_percent(fp, total_time)
        }
    summary.update(summary_part)
    return summary


def calculate_percent(partial, total):
    """Calculate percent value."""
    if total:
        percent = round(partial / total * 100, 2)
    else:
        percent = 0
    return f'{percent}%'


def to_millisecond(sys_count, limit=4):
    """Translate system count to millisecond."""
    per_ms_syscnt = 100000
    return round(sys_count / per_ms_syscnt, limit)


def get_field_value(row_info, field_name, header, time_type='realtime'):
    """
    Extract basic info through row_info.

    Args:
        row_info (list): The list of data info in one row.
        field_name (str): The name in header.
        header (list[str]): The list of field names.
        time_type (str): The type of value, `realtime` or `systime`. Default: `realtime`.

    Returns:
        dict, step trace info in dict format.
    """
    field_index = header.index(field_name)
    value = row_info[field_index]
    value = to_int(value, field_name)
    if time_type == 'realtime':
        value = to_millisecond(value)

    return value


def get_options(options):
    if options is None:
        options = {}

    return options


def combine_stream_task_id(stream_id, task_id):
    """Combine Stream ID and task ID into unique values."""
    return f'{stream_id}_{task_id}'
