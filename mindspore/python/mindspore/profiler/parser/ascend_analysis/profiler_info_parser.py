# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Profiler host information parser"""
import os
import re
import ast
import time
import json
from json import JSONDecodeError
from configparser import ConfigParser
from decimal import Decimal
from typing import Union
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import Popen, PIPE

from mindspore import log as logger
import mindspore._c_expression as c_expression
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.constant import Constant
from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager


class ProfilerInfoParser:
    """Parse files that record information, such as profiler_info.json"""

    _localtime_diff = Decimal(0)
    _syscnt_enable = False
    _freq = 100.0
    _time_offset = 0
    _start_cnt = 0
    _msprof_cmd = "msprof"
    _time_out = 1
    # profiler information related files
    _source_prof_path = None
    _host_start = "host/host_start.log"
    _start_info = "host/start_info"
    _info_json = "host/info.json"

    @classmethod
    def init_source_path(cls, source_path: str):
        """initialize the path of PROF_* directory."""
        source_path = validate_and_normalize_path(source_path)
        prof_path = os.path.dirname(source_path)
        dir_name = os.path.basename(source_path)
        if not dir_name.startswith("device") or not os.path.exists(source_path):
            raise RuntimeError("Input source path is invalid!")
        cls._source_prof_path = prof_path

    @classmethod
    def get_local_time(cls, syscnt: int) -> Decimal:
        """Convert syscnt to local time."""
        if not cls._source_prof_path:
            error_msg = "The path of PROF_* is not initialized!"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        cls.__load_timediff_info()
        cls.__load_syscnt_info()
        start_ns = cls.__get_timestamp(syscnt)
        start_us = Decimal(start_ns) / Constant.NS_TO_US
        local_time = start_us + cls._localtime_diff
        return local_time

    @classmethod
    def get_local_time_new(cls, syscnt: int) -> Decimal:
        """Convert syscnt to local time."""
        syscnt_stamp = c_expression.get_syscnt()
        localtime_stamp = int(time.time() * 1e6)
        outs, _ = cls.__run_cmd(['which', cls._msprof_cmd])
        if not outs:
            raise FileNotFoundError("Failed to find msprof command!")
        msprof_path = os.path.realpath(outs.strip())
        sup_path = msprof_path.split('tools')[0]
        script_path = os.path.join(sup_path, 'tools/profiler/profiler_tool/analysis/interface/get_msprof_info.py')
        py_cmd = ['python', script_path, '-dir', os.path.join(cls._source_prof_path, 'host')]
        outs, _ = cls.__run_cmd(py_cmd)
        if not outs:
            raise RuntimeError("Failed to get msprof information!")
        result = json.loads(outs)
        cpu_info = result.get('data', {}).get('host_info', {}).get('cpu_info', [{}])[0]
        cls._freq = float(cpu_info.get("Frequency", cls._freq))
        cls._start_cnt = syscnt_stamp
        cls._time_offset = localtime_stamp
        start_ns = cls.__get_timestamp(syscnt)
        start_us = Decimal(start_ns) / Constant.NS_TO_US
        return start_us

    @classmethod
    def __run_cmd(cls, cmd):
        """run shell command"""
        try:
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        except (FileNotFoundError, PermissionError, CalledProcessError) as exc:
            raise RuntimeError(exc) from exc
        try:
            outs, errs = proc.communicate(timeout=cls._time_out)
        except TimeoutExpired as err:
            proc.kill()
            msg = "The possible cause is that too much data is collected " \
                  "and the export time is too long."
            logger.error(msg)
            raise TimeoutError(msg) from err
        logger.info(outs)
        return outs, errs

    @classmethod
    def __is_number(cls, number: Union[int, float, str]):
        """Judge the object is number or not."""
        if isinstance(number, (int, float)):
            return True
        pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
        return bool(pattern.match(number))

    @classmethod
    def __load_timediff_info(cls):
        """Update the value of _localtime_diff"""
        start_info_path = os.path.join(cls._source_prof_path, cls._start_info)
        start_info_path = validate_and_normalize_path(start_info_path)
        start_info = ast.literal_eval(FileManager.read_file_content(start_info_path, "rt"))
        # The unit of _localtime_diff is us
        cls._localtime_diff = Decimal(start_info.get(Constant.CANN_BEGIN_TIME, 0)) - Decimal(
            start_info.get(Constant.CANN_BEGIN_MONOTONIC, 0)) / Constant.NS_TO_US

    @classmethod
    def __load_syscnt_info(cls):
        """Update _syscnt_enable, _time_offset and _start_cnt value."""
        # update cls._syscnt_enable
        info_json_path = os.path.join(cls._source_prof_path, cls._info_json)
        info_json_path = validate_and_normalize_path(info_json_path)
        try:
            jsondata = json.loads(FileManager.read_file_content(info_json_path, "rt"))
            config_freq = jsondata.get("CPU")[0].get("Frequency")
            if config_freq is None or not cls.__is_number(config_freq):
                raise ValueError("Do not get valid CPU frequency!")
        except (AttributeError, IndexError, TypeError, ValueError, JSONDecodeError) as err:
            msg = f"Incorrect file content in {os.path.basename(info_json_path)}"
            raise RuntimeError(msg) from err
        if isinstance(config_freq, str) and config_freq.find(".") != -1:
            cls._syscnt_enable = True
        # update cls._time_offset and cls._start_cnt
        cls._freq = float(config_freq)
        host_start_path = os.path.join(cls._source_prof_path, cls._host_start)
        host_start_path = validate_and_normalize_path(host_start_path)
        if not os.path.isfile(host_start_path):
            raise RuntimeError(f"The file {os.path.basename(host_start_path)} is invalid!")
        host_start_log = ConfigParser()
        host_start_log.read(host_start_path)
        config_time_offset = host_start_log.get("Host", "clock_monotonic_raw")
        config_start_cnt = host_start_log.get("Host", "cntvct")
        if cls.__is_number(config_time_offset) and cls.__is_number(config_start_cnt):
            cls._time_offset = int(config_time_offset)
            cls._start_cnt = int(config_start_cnt)

    @classmethod
    def __get_timestamp(cls, syscnt: int, time_fmt: int = 1000):
        """Convert syscnt to time stamp."""
        if not cls._syscnt_enable:
            return syscnt
        ratio = time_fmt / cls._freq
        # The unit of timestamp is ns
        timestamp = round((syscnt - cls._start_cnt) * ratio) + cls._time_offset
        return timestamp
