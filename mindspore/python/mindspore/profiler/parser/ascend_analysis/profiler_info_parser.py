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
import json
from decimal import Decimal
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import Popen, PIPE

from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.constant import Constant
from mindspore.profiler.parser.profiler_info import ProfilerInfo


class ProfilerInfoParser:
    """Parse files that record information, such as profiler_info.json"""

    _freq = 100.0
    _time_offset = 0
    _start_cnt = 0
    _msprof_cmd = "msprof"
    _time_out = 1
    # profiler information related files
    _source_prof_path = None
    _loaded_frequency = False
    _rank_id = 0

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
    def init_rank_id(cls, rank_id: int):
        """initialize the rank id."""
        cls._rank_id = rank_id

    @classmethod
    def get_local_time(cls, syscnt: int) -> Decimal:
        """Convert syscnt to local time."""
        if not cls._loaded_frequency:
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
            try:
                cls._freq = float(cpu_info.get("Frequency", cls._freq))
            except ValueError:
                pass
            profiler_info_path = os.path.join(cls._source_prof_path, os.path.pardir,
                                              f"profiler_info_{cls._rank_id}.json")
            if not os.path.isfile(profiler_info_path):
                raise RuntimeError(f"Can`t find the file {profiler_info_path}, please check !")
            with os.fdopen(os.open(profiler_info_path, os.O_RDONLY, 0o600),
                           'r') as fr:
                profiler_info_data = json.load(fr)
            cls._start_cnt = profiler_info_data.get('system_cnt')
            cls._time_offset = profiler_info_data.get('system_time')
            ProfilerInfo.set_system_time(cls._time_offset)
            ProfilerInfo.set_system_cnt(cls._start_cnt)
            cls._loaded_frequency = True
        start_ns = cls.__get_timestamp(syscnt)
        start_us = Decimal(start_ns * Constant.NS_TO_US)
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
    def __get_timestamp(cls, syscnt: int, time_fmt: int = 1000):
        """Convert syscnt to time stamp."""
        ratio = time_fmt / cls._freq
        # The unit of timestamp is ns
        timestamp = round((syscnt - cls._start_cnt) * ratio) + cls._time_offset
        return timestamp
