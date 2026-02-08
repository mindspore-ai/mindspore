# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""Profiler Path Manager"""
import os
import socket
import glob
import shutil
from datetime import datetime, timezone

from mindspore import log as logger
from mindspore.profiler.common.singleton import Singleton
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.common.path_manager import PathManager


@Singleton
class ProfilerPathManager:
    """
    ProfilerPathManager is responsible for creating and managing all paths used by profiler.
    """

    _ASCEND_MS_DIR = "{}_{}_ascend_ms"
    MAX_WORKER_NAME_LENGTH = 226

    def __init__(self):
        self._prof_ctx = ProfilerContext()
        self._worker_name = None
        self._dir_path = None

    def init(self, worker_name: str = None, dir_name: str = None) -> None:
        """
        Init the profiler path.
        """
        valid_wk_name = worker_name and isinstance(worker_name, str)
        valid_wk_len = isinstance(worker_name, str) and len(worker_name) < self.MAX_WORKER_NAME_LENGTH
        if (valid_wk_name and valid_wk_len) or worker_name is None:
            self._worker_name = worker_name
        else:
            logger.warning("Invalid parameter worker_name, reset it to default.")
            self._worker_name = None

        valid_dir_name = dir_name and isinstance(dir_name, str)
        if valid_dir_name:
            dir_path = PathManager.get_real_path(dir_name)
            PathManager.check_input_directory_path(dir_path)
            self._dir_path = dir_path
        elif dir_name is None:
            self._dir_path = dir_name
        else:
            logger.warning(f"Invalid parameter dir_name, reset it to default.")
            self._dir_path = None

        if self._dir_path:
            self._prof_ctx.on_trace_ready_output_path = self._dir_path

    def clean_analysis_cache(self):
        """
        Clean the profiler analysis cache.
        """
        ANALYSIS_CACHE = (
            # ASEND_PROFILER_OUTPUT_PATH
            self._prof_ctx.ascend_profiler_output_path,
            # PROF_XXX/mindstudio_profiler_output
            self._prof_ctx.msprof_profile_output_path,
            # PROF_XXX/mindstudio_profiler_log
            self._prof_ctx.msprof_profile_log_path,
            # PROF_XXX/host/sqlite
            os.path.join(self._prof_ctx.msprof_profile_host_path, "sqlite"),
            # PROF_XXX/host/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_host_path, "data", "all_file.complete"),
            # PROF_XXX/device_x/sqlite
            os.path.join(self._prof_ctx.msprof_profile_device_path, "sqlite"),
            # PROF_XXX/device_x/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_device_path, "data", "all_file.complete"),
        )

        for cache_path in ANALYSIS_CACHE:
            if os.path.isfile(cache_path):
                PathManager.remove_file_safety(cache_path)
            elif os.path.isdir(cache_path):
                PathManager.remove_path_safety(cache_path)

    def simplify_data(self):
        """
        Simplify the profiler data.
        """
        SIMPLIFY_CACHE = (
            # PROF_XXX/mindstudio_profiler_output
            self._prof_ctx.msprof_profile_output_path,
            # PROF_XXX/mindstudio_profiler_log
            self._prof_ctx.msprof_profile_log_path,
            # PROF_XXX/host/sqlite
            os.path.join(self._prof_ctx.msprof_profile_host_path, "sqlite"),
            # PROF_XXX/host/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_host_path, "data", "all_file.complete"),
            # PROF_XXX/device_x/sqlite
            os.path.join(self._prof_ctx.msprof_profile_device_path, "sqlite"),
            # PROF_XXX/device_x/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_device_path, "data", "all_file.complete"),
        )

        for cache_path in SIMPLIFY_CACHE:
            if os.path.isfile(cache_path):
                PathManager.remove_file_safety(cache_path)
            elif os.path.isdir(cache_path):
                PathManager.remove_path_safety(cache_path)

    def move_db_file(self):
        """
        Copy the db file to the output path.
        """
        if not self._prof_ctx.msprof_profile_output_path:
            return
        db_files = glob.glob(os.path.join(
            os.path.dirname(self._prof_ctx.msprof_profile_output_path),
            'msprof*.db'
        )) + glob.glob(os.path.join(
            os.path.dirname(self._prof_ctx.msprof_profile_output_path),
            "analyze",
            "communication_analyzer.db"
        ))
        for db_file in db_files:
            if os.path.isfile(db_file):
                db_file_name = os.path.basename(db_file)
                if db_file_name == "communication_analyzer.db":
                    new_file_name = os.path.join(self._prof_ctx.ascend_profiler_output_path, db_file_name)
                    shutil.copy(db_file, new_file_name)
                else:
                    new_file_name = f"ascend_mindspore_profiler_{self._prof_ctx.rank_id}.db" if self._prof_ctx.rank_id \
                        else f"ascend_mindspore_profiler.db"
                    new_file_path = os.path.join(self._prof_ctx.ascend_profiler_output_path, new_file_name)
                    shutil.move(db_file, new_file_path)

    def remove_db_file(self):
        """
        Remove the db file in the output path.
        """
        if not self._prof_ctx.msprof_profile_output_path:
            return
        db_files = glob.glob(os.path.join(
            os.path.dirname(self._prof_ctx.msprof_profile_output_path),
            'msprof*.db'
        ))
        for db_file in db_files:
            if os.path.isfile(db_file):
                os.remove(db_file)

    def create_output_path(self):
        """
        Create ASCEND_PROFILER_OUTPUT dir, this method should call before analysis
        """
        PathManager.make_dir_safety(self._prof_ctx.ascend_profiler_output_path)

    def set_ascend_ms_dir(self):
        """
        reset xxx_ascend_ms name
        """
        self._prof_ctx.ascend_ms_dir = self._get_ascend_ms_dir()

    def create_profiler_paths(self):
        """
        Create xxx_ascend_ms and FRAMEWORK dir, this method should call before Profiler start
        """
        PathManager.make_dir_safety(self._prof_ctx.ascend_ms_dir)
        PathManager.make_dir_safety(self._prof_ctx.framework_path)
        logger.info(
            "Profiler ascend_ms_dir initialized: %s", self._prof_ctx.ascend_ms_dir
        )

    def _get_ascend_ms_dir(self) -> str:
        """
        Generate xxx_ascend_ms name
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
        if not self._worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        else:
            worker_name = f"{self._worker_name}_{os.getpid()}"

        return self._ASCEND_MS_DIR.format(worker_name, timestamp)
