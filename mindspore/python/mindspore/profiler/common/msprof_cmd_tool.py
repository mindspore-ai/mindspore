# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Tool for managing Ascend msprof profiling commands and environment."""
import os
import json
import shutil
from functools import lru_cache
from typing import Dict, List, Optional

from mindspore import log as logger
from mindspore.profiler.common.command_executor import CommandExecutor
from mindspore.profiler.common.constant import ExportType
from mindspore.profiler.common.path_manager import PathManager


class MsprofCmdTool:
    """Tool for managing Ascend msprof profiling commands and environment."""

    _MSPROF_CMD = "msprof"
    _ASCEND_MARK = "Ascend"
    _HIAI_MSPROF_TAIL = "Ascend/cann/tools/profiler/bin"
    _MSPROF_PY_PATH = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
    _MSPROF_INFO_PATH = "tools/profiler/profiler_tool/analysis/interface/get_msprof_info.py"

    def __init__(self, msprof_profile_path: str):
        """Initialize MsprofTool."""
        self._msprof_profile_path = msprof_profile_path
        self._check_environment()

    def run_ms_export_cmd(self, analyse_export_type: List[str]) -> None:
        """Run msprof export command.

        Args:
            analyse_export_type (List[str]): The type of data to export.
        """
        export_cmd = [
            self._MSPROF_CMD,
            "--export=on",
            f"--output={self._msprof_profile_path}",
        ]
        if ExportType.Text.value in analyse_export_type:
            CommandExecutor.execute(export_cmd)
        if ExportType.Db.value in analyse_export_type:
            export_cmd.append("--type=db")
            CommandExecutor.execute(export_cmd)

    def run_ms_py_export_cmd(self, model_id: int, iter_list: List[int], analyse_export_type: List[str]) -> None:
        """Export timeline and summary data for the specified model and iterations.

        Args:
            model_id (int): The ID of the model to export data for.
            iter_list (List[int]): A list of iteration IDs to export data for.
            analyse_export_type (List[str]): The type of data to export.

        Raises:
            FileNotFoundError: If msprof.py path cannot be found.
        """
        script_path = self._get_msprof_script_path(self._MSPROF_PY_PATH)
        if not script_path:
            raise FileNotFoundError(
                "Failed to find msprof.py path. Please check the CANN environment."
            )

        export_cmd = ["python3", script_path]
        iter_param = self._get_iteration_params(model_id, iter_list)

        if ExportType.Text.value in analyse_export_type:
            for export_type in ("timeline", "summary"):
                cmd = (
                    export_cmd
                    + ["export", export_type, "-dir", self._msprof_profile_path]
                    + iter_param
                )
                CommandExecutor.execute(cmd)
        if ExportType.Db.value in analyse_export_type:
            cmd = (
                export_cmd
                + ["export", "db", "-dir", self._msprof_profile_path]
            )
            CommandExecutor.execute(cmd)

    def run_ms_analyze_cmd(self, analyse_export_type: List[str]) -> None:
        """Run msprof analyze command."""
        analyze_cmd = [
            self._MSPROF_CMD,
            "--analyze=on",
            "--rule=communication,communication_matrix",
            f"--output={self._msprof_profile_path}",
        ]
        if ExportType.Text.value in analyse_export_type:
            CommandExecutor.execute(analyze_cmd)
        if ExportType.Db.value in analyse_export_type:
            analyze_cmd.append("--type=db")
            CommandExecutor.execute(analyze_cmd)

    @lru_cache(maxsize=1)
    def get_msprof_info(self):
        """Get the msprof info.

        Returns:
            Dict: Msprof information dictionary.
        """
        msprof_info = self._setup_msprof_info()
        return msprof_info

    def _check_environment(self) -> None:
        """Check if required commands are available in the environment.

        Raises:
            FileNotFoundError: If msprof or python3 command is not found.
        """
        self._check_msprof_profile_path_is_valid()
        if not shutil.which(self._MSPROF_CMD):
            logger.warning(
                "The msprof command is not found in PATH. Searching in environment variables..."
            )
            msprof_path = self._find_msprof_path()

            if msprof_path:
                os.environ["PATH"] = f"{msprof_path}:{os.environ.get('PATH', '')}"
                logger.info("Successfully added msprof command to PATH.")
            else:
                raise FileNotFoundError("Failed to find msprof command in environment.")
        else:
            msprof_path = shutil.which(self._MSPROF_CMD)
        self._check_msprof_permission(msprof_path)
        if not shutil.which("python3"):
            logger.warning("Failed to find python3 command in environment.")
            raise FileNotFoundError("Failed to find python3 command in environment.")

    def _check_msprof_profile_path_is_valid(self):
        """Check msprof profiler path is invalid."""
        PathManager.check_directory_path_readable(self._msprof_profile_path)
        PathManager.check_directory_path_writeable(self._msprof_profile_path)
        PathManager.check_path_owner_consistent(self._msprof_profile_path)
        PathManager.check_path_is_other_writable(self._msprof_profile_path)
        if not PathManager.check_path_is_executable(self._msprof_profile_path):
            raise PermissionError(f"The '{self._msprof_profile_path}' path is not executable."
                                  f"Please execute chmod -R 755 {self._msprof_profile_path}")

    def _check_msprof_permission(self, msprof_path):
        """Check msprof path permissions."""
        msprof_script_path = self._get_msprof_script_path(self._MSPROF_PY_PATH)
        if not msprof_script_path:
            raise FileNotFoundError(
                "Failed to find msprof.py path. Perhaps the permission of the 'msprof' tool is unexecutable. "
                "Please check the CANN environment. You can modify the 'msprof' file to an executable permission "
                "through the chmod method."
            )
        if not PathManager.check_path_is_owner_or_root(msprof_script_path) or \
                not PathManager.check_path_is_owner_or_root(msprof_path):
            raise PermissionError(f"PermissionError, CANN package user id: {os.stat(msprof_path).st_uid}, "
                                  f"current user id: {os.getuid()}. "
                                  f"Ensure CANN package user id and current user id consistency")
        if not PathManager.check_path_is_executable(msprof_script_path) or \
                not PathManager.check_path_is_executable(msprof_path):
            raise PermissionError(f"The '{msprof_script_path}' path or '{msprof_path}' path is not executable."
                                  f"Please execute chmod u+x {msprof_script_path} and "
                                  f"chmod u+x {msprof_path}")
        PathManager.check_path_is_other_writable(msprof_script_path)

    def _find_msprof_path(self) -> Optional[str]:
        """Find msprof path in environment variables.

        Returns:
            Optional[str]: Path to msprof if found, None otherwise.
        """
        if os.environ.get("ASCEND_TOOLKIT_HOME"):
            temp_path = os.path.join(os.environ.get("ASCEND_TOOLKIT_HOME"), "bin")
            if os.path.isdir(temp_path) and self._MSPROF_CMD in os.listdir(temp_path):
                return os.path.abspath(temp_path)

        for path in os.environ.get("PATH", "").split(":"):
            if self._ASCEND_MARK in path:
                prefix = path.split(self._ASCEND_MARK)[0]
                temp_path = os.path.join(prefix, self._HIAI_MSPROF_TAIL)
                if os.path.isdir(temp_path) and self._MSPROF_CMD in os.listdir(temp_path):
                    return os.path.abspath(temp_path)

        return None

    def _setup_msprof_info(self) -> Dict:
        """Get msprof information.

        Returns:
            Dict: Msprof information dictionary.
        """
        script_path = self._get_msprof_script_path(self._MSPROF_INFO_PATH)
        if not script_path:
            logger.error("Failed to find get_msprof_info.py path.")
            return {}
        if not PathManager.check_path_is_executable(script_path):
            raise PermissionError(f"The '{script_path}' path is not executable. Please execute chmod u+x {script_path}")
        host_dir = os.path.join(self._msprof_profile_path, "host")
        cmd = ["python3", script_path, "-dir", host_dir]
        command_outs = CommandExecutor.execute(cmd)[0]

        try:
            return json.loads(command_outs)
        except json.JSONDecodeError as err:
            logger.error(f"Failed to decode msprof info. Error: {str(err)}")
            return {}

    def _get_msprof_script_path(self, script_path: str) -> str:
        """Get the full path of a msprof script.

        Args:
            script_path (str): Relative path to the script.

        Returns:
            str: Full path to the script if found, empty string otherwise.
        """
        msprof_path = shutil.which(self._MSPROF_CMD)
        if not msprof_path:
            return ""
        msprof_path = os.path.realpath(msprof_path.strip())
        pre_path = msprof_path.split("tools")[0]
        full_script_path = os.path.join(pre_path, script_path)
        return full_script_path if os.path.exists(full_script_path) else ""

    @staticmethod
    def _get_iteration_params(model_id: int, iter_list: List[int]) -> List[str]:
        """Get iteration parameters for msprof.py script.

        Args:
            model_id (int): Model ID.
            iter_list (List[int]): List of iteration IDs.

        Returns:
            List[str]: List of parameter strings.
        """
        iter_param = []
        if isinstance(model_id, int) and model_id >= 0:
            iter_param.extend(["--model-id", str(model_id)])
        if iter_list and isinstance(iter_list, list):
            iter_list.sort()
            iter_param.extend(
                [
                    "--iteration-id",
                    str(iter_list[0]),
                    "--iteration-count",
                    str(len(iter_list)),
                ]
            )
        return iter_param
