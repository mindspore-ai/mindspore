# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""ProfilerInterface"""
from typing import Set

from mindspore import log as logger
from mindspore.graph.api import _pynative_executor
from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.platform.base_profiler import BaseProfiler
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager
from mindspore.profiler.common.profiler_meta_data import ProfilerMetaData


class ProfilerInterface:
    """
    Profiler interface
    """

    cpu_profiler: BaseProfiler = None
    device_profiler: BaseProfiler = None
    platform_profilers_set: Set[BaseProfiler] = set()
    is_initialized = False

    @classmethod
    def init(cls):
        """ProfilerInterface init"""
        if cls.is_initialized:
            return

        ProfilerPathManager().set_ascend_ms_dir()
        ProfilerPathManager().create_profiler_paths()
        platforms = ProfilerContext().device_target_set
        for device_target in platforms:
            cls.platform_profilers_set.add(PROFILERS.modules.get(device_target)())

        cls.is_initialized = True
        logger.info("ProfilerInterface init")

    @classmethod
    def start(cls):
        """ProfilerInterface start"""
        if not cls.is_initialized:
            logger.warning("ProfilerInterface start failed, profiler has not been initialized.")
            return

        for profiler in cls.platform_profilers_set:
            profiler.start()

        logger.info("ProfilerInterface start")

    @classmethod
    def stop(cls):
        """ProfilerInterface stop"""
        if not cls.is_initialized:
            logger.warning("ProfilerInterface stop failed, profiler has not been initialized.")
            return

        _pynative_executor.sync()
        for profiler in cls.platform_profilers_set:
            profiler.stop()

        logger.info("ProfilerInterface stop")

    @classmethod
    def analyse(cls, **kwargs):
        """ProfilerInterface analyse"""
        if not cls.is_initialized:
            logger.warning("ProfilerInterface analyse failed, profiler has not been initialized.")
            return

        for profiler in cls.platform_profilers_set:
            profiler.analyse(**kwargs)

        logger.info("ProfilerInterface analyse")

    @classmethod
    def finalize(cls):
        """ProfilerInterface finalize"""
        if not cls.is_initialized:
            logger.warning("ProfilerInterface finalize failed, profiler has not been initialized.")
            return

        ProfilerMetaData.dump_metadata()
        for profiler in cls.platform_profilers_set:
            profiler.finalize()
            profiler = None

        logger.info("ProfilerInterface finalize")

    @classmethod
    def clear(cls):
        """ProfilerInterface clear"""
        if not cls.is_initialized:
            logger.warning("ProfilerInterface clear failed, profiler has not been initialized.")
            return
        cls.platform_profilers_set.clear()
        cls.is_initialized = False
        ProfilerLogger.destroy()
        logger.info("ProfilerInterface clear")

    @classmethod
    def delete_dir(cls):
        """ProfilerInterface delete dir"""
        logger.info("ProfilerInterface delete dir")
