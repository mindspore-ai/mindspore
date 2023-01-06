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
import json
import os
import time
from enum import Enum
from mindspore.profiler import Profiler
from mindspore.profiler.profiling import AICORE_METRICS_DICT
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.integrator import DeviceTarget

from mindspore import log as logger, context


class DeviceSupportParam(Enum):
    """The device target enum."""
    CPU = ['start', 'output_path']
    GPU = ['start', 'output_path', 'data_process', 'timeline_limit', 'sync_enable']
    ASCEND = ['start', 'output_path', 'data_process', 'timeline_limit', 'profile_memory', 'parallel_strategy',
              'profile_communication', 'aicore_metrics', 'l2_cache']


def get_profiling_options():
    """Get profiling options."""
    try:
        options = json.loads(os.environ.get("MS_PROFILER_OPTIONS", "{}"))
    except json.JSONDecodeError:
        return None
    return options


def parse_device_support_param(options):
    """Parse platform support parameters."""
    device_target = context.get_context("device_target").upper()
    for param in options.keys():
        if param not in DeviceSupportParam.__getattr__(f'{device_target}').value:
            logger.warning(f"The parameter '{param}' is not supported on {device_target} currently.")


def construct_profiling_options():
    """Construct profiling options to determine which profiling data should be collected."""
    profiling_options = get_profiling_options()
    if profiling_options is None:
        raise RuntimeError(
            "The format of MS_PROFILER_OPTIONS is incorrect. "
            "The MS_PROFILER_OPTIONS parameter configuration may refer to "
            "'https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html'."
        )
    options = combine_profile_options(profiling_options)
    conbine_options = parse_profiling_args(options)
    if conbine_options.get("start"):
        parse_device_support_param(profiling_options)
        output_path = conbine_options.get("output_path")
        if not output_path:
            output_path = os.path.join(os.getcwd(), "data")
        conbine_options["output_path"] = validate_and_normalize_path(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        conbine_options["file_output_path"] = os.path.join(output_path, "profiler")
    return conbine_options


def parse_pubilc_args(options):
    """Parsing three platform profiling args."""
    if not isinstance(options.get("start"), bool):
        logger.warning(
            "The 'start' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('start'))}, it will be set to false.")
        options["start"] = False
    if not options.get("start"):
        return options
    if not isinstance(options.get("data_process"), bool):
        logger.warning(
            "The 'data_process' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('data_process'))}, it will be set to true.")
        options["data_process"] = True
    if not isinstance(options.get("timeline_limit"), int):
        logger.warning(
            "The 'timeline_limit' parameter of the environment variable MS_PROFILE_OPTIONS must be int,"
            f" but got type {type(options.get('timeline_limit'))}, it will be set to 500.")
        options["timeline_limit"] = 500
    absolute_path = os.path.join(os.getcwd(), "data")
    if not isinstance(options.get("output_path"), str):
        logger.warning(
            "The 'output_path' parameter of the environment variable MS_PROFILE_OPTIONS must be str,"
            f" but got type {type(options.get('output_path'))}, it will be set to '{absolute_path}'.")
        options["output_path"] = absolute_path
    if not os.path.isabs(options.get("output_path")):
        logger.warning(
            "The 'output_path' parameter of the environment variable MS_PROFILE_OPTIONS only supports absolute path, "
            f"it will be set to '{absolute_path}'.")
        options["output_path"] = absolute_path
    return options


def parse_gpu_args(options):
    """Parsing gpu profiling args."""
    if not isinstance(options.get("sync_enable"), bool):
        logger.warning(
            "The 'sync_enable' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('sync_enable'))}, it will be set to true.")
        options["sync_enable"] = True
    return options


def parse_ascend_args(options):
    """Parsing ascend profiling args."""
    if not isinstance(options.get("profile_memory"), bool):
        logger.warning(
            "The 'profile_memory' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('profile_memory'))}, it will be set to false.")
        options["profile_memory"] = False
    if not isinstance(options.get("parallel_strategy"), bool):
        logger.warning(
            "The 'parallel_strategy' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('parallel_strategy'))}, it will be set to true.")
        options["parallel_strategy"] = True
    if not isinstance(options.get("profile_communication"), bool):
        logger.warning(
            "The 'profile_communication' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('profile_communication'))}, it will be set to false.")
        options["profile_communication"] = False
    if options.get("aicore_metrics") not in AICORE_METRICS_DICT:
        logger.warning(
            "The 'aicore_metrics' parameter of the environment variable MS_PROFILE_OPTIONS must be in "
            f"[-1, 0, 1, 2, 3, 4, 5], but got {options.get('aicore_metrics')}, it will be set to 0.")
        options["aicore_metrics"] = 0
    if not isinstance(options.get("l2_cache"), bool):
        logger.warning(
            "The 'l2_cache' parameter of the environment variable MS_PROFILE_OPTIONS must be bool,"
            f" but got type {type(options.get('l2_cache'))}, it will be set to false.")
        options["l2_cache"] = False
    return options


def parse_profiling_args(options):
    """Parsing profiling args."""
    profiling_options = parse_pubilc_args(options)
    if not profiling_options.get("start"):
        return profiling_options
    if context.get_context("device_target").lower() == DeviceTarget.ASCEND.value:
        options = parse_ascend_args(profiling_options)
    if context.get_context("device_target").lower() == DeviceTarget.GPU.value:
        options = parse_gpu_args(profiling_options)
    return options


def combine_profile_options(profiling_options):
    """Combined profiling options."""
    factor_s_to_us = 1e7
    output_path = os.path.join(os.getcwd(), "data")
    if context.get_context("device_target").upper() == "GPU":
        sync_enable = profiling_options.get("sync_enable", True)
    else:
        sync_enable = profiling_options.get("sync_enable", False)
    options = {
        "start": profiling_options.get('start', False),
        "start_time": int(time.time() * factor_s_to_us),
        "pid": os.getpid(),
        "output_path": profiling_options.get('output_path', output_path),
        "file_output_path": "",
        "profile_memory": profiling_options.get("profile_memory", False),
        "profile_communication": profiling_options.get("profile_communication", False),
        "aicore_metrics": profiling_options.get("aicore_metrics", 0),
        "l2_cache": profiling_options.get("l2_cache", False),
        "sync_enable": sync_enable,
        "data_process": profiling_options.get("data_process", True),
        "timeline_limit": profiling_options.get("timeline_limit", 500),
        "parallel_strategy": profiling_options.get("parallel_strategy", True),
    }
    return options


class EnvProfiler:
    """Collect and analyze training performance data, support calls during and after training."""

    def __init__(self):
        self._profiling_options = {}

    def analyse(self):
        """Determine whether to stop collecting and parsing performance data based on environment variables."""
        if not os.getenv("MS_PROFILER_OPTIONS"):
            return
        self._profiling_options = json.loads(os.getenv("MS_PROFILER_RUN_CONFIG", "{}"))
        if not self._profiling_options.get("pid", 0) == os.getpid():
            return
        if not self._profiling_options.get("start"):
            return
        profiler = Profiler(env_enable=self._profiling_options)
        profiler.analyse()


def profiler_check_env():
    """Profiler initialization according to environment."""
    if not os.getenv("MS_PROFILER_OPTIONS"):
        return
    if os.getenv("MS_PROFILER_RUN_CONFIG"):
        return
    config = construct_profiling_options()
    os.environ["MS_PROFILER_RUN_CONFIG"] = json.dumps(config)
    if not config.get("start"):
        return
    Profiler(output_path=config.get("output_path"),
             profile_memory=config.get("profile_memory"),
             profile_communication=config.get("profile_communication"),
             data_process=config.get("data_process"),
             parallel_strategy=config.get("parallel_strategy"),
             aicore_metrics=config.get("aicore_metrics"),
             l2_cache=config.get("l2_cache"),
             sync_enable=config.get("sync_enable"),
             timeline_limit=config.get("timeline_limit"))


profiler_check_env()
