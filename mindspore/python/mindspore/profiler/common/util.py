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
import time
from functools import wraps
from typing import Any, Callable, Type, Tuple, Union
import re
import shutil
import stat

from mindspore.communication.management import GlobalComm
from mindspore.communication.management import get_local_rank
from mindspore import context
from mindspore import log as logger
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.exceptions.exceptions import ProfilerPathErrorException

from mindspore.profiler.common.constant import DeviceTarget


def no_exception_func(
        default_ret: Any = None,
        exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        custom_handler: Callable[[Exception], Any] = None
) -> Callable[[Callable], Callable]:
    """
    A decorator that catches specified exceptions and handles them gracefully.

    Args:
        default_ret (Any, optional): The default return value if an exception occurs. Defaults to None.
        exception_types (Union[Type[Exception], Tuple[Type[Exception], ...]], optional):
            The type(s) of exceptions to catch. Defaults to Exception.
        custom_handler (Callable[[Exception], Any], optional):
            A custom function to handle the caught exception. Defaults to None.

    Returns:
        Callable[[Callable], Callable]: A decorator function.

    Example:
        @no_exception_func(default_ret=0, exception_types=(ValueError, TypeError))
        def divide(a, b):
            return a / b
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exception_types as ex: # pylint: disable=W0703
                if custom_handler:
                    return custom_handler(ex)

                logger.error(f"Call to {func.__name__} failed. Exception: {str(ex)}")
                return default_ret
        return wrapper
    return decorator


def get_cann_version():
    """
    get cann version from env
    """
    ascend_home_path = os.environ.get("ASCEND_HOME_PATH", "")
    cann_version = "not known"
    try:
        PathManager.check_path_is_readable(os.path.realpath(ascend_home_path))
        for dirpath, _, filenames in PathManager.walk_with_depth(os.path.realpath(ascend_home_path)):
            install_files = [file for file in filenames if re.match(r"ascend_.{1,20}_install\.info", file)]
            if install_files:
                filepath = os.path.realpath(os.path.join(dirpath, install_files[0]))
                PathManager.check_path_is_readable(filepath)
                with open(filepath, "r") as f:
                    for line in f:
                        if line.find("version") != -1:
                            cann_version = line.strip().split("=")[-1]
                            break
    except ProfilerPathErrorException as e:
        logger.warning(f"Failed to read CANN version from {ascend_home_path}: {e}")
    return cann_version


def timeit(custom_message=None):
    """
    Decorator to measure the execution time of a function.

    Args:
        custom_message (str, optional): Custom message to display. If not provided, the function name will be used.

    Returns:
        The result of the decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time

            message = custom_message or func.__name__
            print(f"[{os.getpid()}] {message}, cost time {execution_time:.2f} s", flush=True)

            return result

        return wrapper

    return decorator


def print_msg_with_pid(msg):
    """
    Print message with process id.

    Args:
        msg (str): The message to print.
    """
    print(f"[{os.getpid()}] {msg}", flush=True)


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


def get_newest_file(file_list):
    """
    Find the newest files
    :param file_list:
    :return:
    """
    newest_file_list = []
    newest_timestamp = '0'
    for file_path in file_list:
        timestamp = file_path.split('.')[0].split('/')[-1].split('_')[-1]
        newest_timestamp = max(timestamp, newest_timestamp)

    for file_path in file_list:
        if file_path.split('.')[0].split('/')[-1].split('_')[-1] == newest_timestamp:
            newest_file_list.append(file_path)

    newest_file_list.sort()
    return newest_file_list


def get_device_id():
    """
    Get device ID.
    """
    device_id = str(context.get_context("device_id"))

    if not device_id or not device_id.isdigit():
        if GlobalComm.INITED and context.get_context("device_target") == DeviceTarget.NPU.value:
            device_id = str(get_local_rank())
        else:
            device_id = os.getenv("DEVICE_ID")

    if not device_id or not device_id.isdigit():
        logger.warning("Fail to get DEVICE_ID, use 0 instead.")
        device_id = "0"
    return device_id


class ProfilerPathManager:
    """A path manager to manage profiler path"""

    FRAMEWORK_DIR = "FRAMEWORK"
    INVALID_VALUE = -1

    @classmethod
    def get_fwk_path(cls, profiler_path: str) -> str:
        """Get FRAMEWORK directory path"""
        fwk_path = os.path.join(profiler_path, cls.FRAMEWORK_DIR)
        if os.path.isdir(fwk_path):
            return fwk_path
        return ""

    @classmethod
    def get_cann_path(cls, profiler_path: str) -> str:
        """Get CANN Prof directory path"""
        sub_dirs = os.listdir(os.path.realpath(profiler_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(profiler_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^PROF_\d+_\d+_[0-9a-zA-Z]+", sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_host_path(cls, cann_path: str) -> str:
        """Get CANN Prof host directory path"""
        host_path = os.path.join(cann_path, 'host')
        if os.path.exists(host_path):
            return host_path
        return ""

    @classmethod
    def get_device_path(cls, cann_path: str) -> str:
        """Get CANN Prof device directory path"""
        sub_dirs = os.listdir(os.path.realpath(cann_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(cann_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^device_\d", sub_dir):
                return sub_path
        return ""

    @classmethod
    def remove_path_safety(cls, path: str):
        """Remove directory"""
        msg = f"Failed to remove path: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def remove_file_safety(cls, file: str):
        """Remove file"""
        msg = f"Failed to remove file: {file}"
        if os.path.islink(file):
            raise RuntimeError(msg)
        if not os.path.exists(file):
            return
        try:
            os.remove(file)
        except FileExistsError:
            return
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def simplify_data(cls, profiler_path: str, simplify_flag: bool):
        """Profiler simplify temporary data"""
        cann_path = cls.get_cann_path(profiler_path)
        device_path = cls.get_device_path(cann_path)
        host_path = cls.get_host_path(cann_path)
        rm_dirs = ['sqlite', 'summary', 'timeline'] if simplify_flag else ['sqlite']
        for rm_dir in rm_dirs:
            if device_path:
                target_path = os.path.join(device_path, rm_dir)
                cls.remove_path_safety(target_path)
            if host_path:
                target_path = os.path.join(host_path, rm_dir)
                cls.remove_path_safety(target_path)
        if simplify_flag:
            fwk_path = cls.get_fwk_path(profiler_path)
            cls.remove_path_safety(fwk_path)
            if not cann_path:
                return
            cann_rm_dirs = ['analyze', 'mindstudio_profiler_log', 'mindstudio_profiler_output']
            for cann_rm_dir in cann_rm_dirs:
                target_path = os.path.join(cann_path, cann_rm_dir)
                cls.remove_path_safety(target_path)
            log_patten = r'msprof_anlysis_\d+\.log$'
            for cann_file in os.listdir(cann_path):
                file_path = os.path.join(cann_path, cann_file)
                if not os.path.isfile(file_path):
                    continue
                if re.match(log_patten, cann_file):
                    cls.remove_file_safety(file_path)
