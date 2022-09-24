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
"""Whether to enable profiler using environment variables."""
import os
import json
import time
from enum import Enum
from mindspore import log as logger, context
from mindspore.profiler import Profiler
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


def construct_profiling_options():
    """Construct profiling options to determine which profiling data should be collected."""
    options = combine_profile_options()
    try:
        profiling_options = json.loads(os.environ.get("MS_PROFILER_OPTIONS", "{}"))
    except json.JSONDecodeError as err:
        raise RuntimeError('The format of PROFILING_OPTIONS is incorrect.') from err
    options["start"] = profiling_options.get('start', False)
    options["profile_memory"] = profiling_options.get("memory", False)
    options["profile_communication"] = profiling_options.get("hccl", False)
    if not isinstance(options.get("start"), bool):
        raise ValueError(
            "The 'start' parameter of the environment variable MS_PROFILE_OPTIONS must be set to true or false.")
    if not isinstance(options.get("profile_memory"), bool):
        raise ValueError(
            "The 'memory' parameter of the environment variable MS_PROFILE_OPTIONS must be set to true or false.")
    if not isinstance(options.get("profile_communication"), bool):
        raise ValueError(
            "The 'hccl' parameter of the environment variable MS_PROFILE_OPTIONS must be set to true or false.")
    if options.get("start"):
        output_path = profiling_options.get("output_path")
        if not output_path:
            output_path = os.path.join(os.getcwd(), "data")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        options["output_path"] = validate_and_normalize_path(output_path)
        options["profiler_path"] = os.path.join(output_path, "profiler")
        return options
    return options


def combine_profile_options():
    """Combined profiling options."""
    factor_s_to_us = 1e7
    options = {
        "start": False,
        "start_time": int(time.time() * factor_s_to_us),
        "pid": os.getpid(),
        "output_path": "",
        "profiler_path": "",
        "profile_memory": False,
        "profile_communication": False
    }
    return options


def get_rank_id_and_target():
    """Get device id and rank id and target of this training."""
    device_target, dev_id, rank_id = "", "", ""
    try:
        dev_id = str(context.get_context("device_id"))
        device_target = context.get_context("device_target").lower()
    except ValueError as err:
        logger.error("Profiling: fail to get context, %s", err)

    if not dev_id or not dev_id.isdigit():
        dev_id = os.getenv('DEVICE_ID')
    if not dev_id or not dev_id.isdigit():
        dev_id = "0"
        logger.warning("Fail to get DEVICE_ID, use 0 instead.")

    if device_target and device_target not in [DeviceTarget.ASCEND.value, DeviceTarget.GPU.value,
                                               DeviceTarget.CPU.value]:
        msg = "Profiling: unsupported backend: %s" % device_target
        raise RuntimeError(msg)

    rank_id = os.getenv("RANK_ID")
    if not rank_id or not rank_id.isdigit():
        rank_id = "0"
    rank_id = rank_id if device_target == DeviceTarget.ASCEND.value else dev_id
    return device_target, rank_id


class DeviceTarget(Enum):
    """The device target enum."""
    CPU = 'cpu'
    GPU = 'gpu'
    ASCEND = 'ascend'


class EnvProfiler:
    """Collect and analyze training performance data, support calls during and after training."""

    def __init__(self):
        self._profiling_options = ''
        self._profiler_manager = None
        self._cpu_profiler = None
        self._md_profiler = None
        self._dynamic_status = False
        self._environ_enable = False
        self._output_path = False
        self._process_name = ""
        self.memory = False
        self.hccl = False
        self.has_end = False
        self.device_target = ""
        self.rank_id = ""
        self.start_time = 0

    def analyse(self):
        """Determine whether to stop collecting and parsing performance data based on environment variables."""
        options = construct_profiling_options()
        self._environ_enable = options.get("start")
        self._output_path = options.get("profiler_path")
        self.memory = options.get("profile_memory")
        self.hccl = options.get("profile_communication")
        if not self._environ_enable:
            return
        env_options = json.loads(os.getenv("MS_PROFILER_RUN_CONFIG", "{}"))
        if not env_options.get("pid", 0) == os.getpid():
            return
        self.device_target, self.rank_id = get_rank_id_and_target()
        self.start_time = env_options.get("start_time")
        options = {
            "output_path": self._output_path,
            "profile_memory": self.memory,
            "profile_communication": self.hccl,
            "start_time": self.start_time
        }
        profiler = Profiler(env_enable=options)
        profiler.analyse()


def profiler_init():
    """Profiler initialization according to environment."""
    if os.getenv("MS_PROFILER_RUN_CONFIG"):
        return
    config = construct_profiling_options()
    if not config.get("start"):
        return
    os.environ["MS_PROFILER_RUN_CONFIG"] = json.dumps(config)
    Profiler(output_path=config.get("output_path"),
             profile_memory=config.get("profile_memory"),
             profile_communication=config.get("profile_communication"))


profiler_init()
