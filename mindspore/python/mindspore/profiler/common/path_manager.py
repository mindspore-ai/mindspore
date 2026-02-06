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
"""Profiler path manager"""
import os
import re
import shutil
import glob
import stat

from mindspore import log as logger
from mindspore.profiler.common.constant import FileConstant
from mindspore.profiler.common.exceptions.exceptions import ProfilerPathErrorException


class PathManager:
    """
    Path common operations manager
    """
    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o750
    MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10

    @classmethod
    def check_input_directory_path(cls, path: str):
        """
        Function Description:
            check whether the path is valid, some businesses can accept a path that does not exist,
            so the function do not verify whether the path exists
        Parameter:
            path: the path to check, whether the incoming path is absolute or relative depends on the business
        Exception Description:
            when invalid data throw exception
        """
        cls._input_path_common_check(path)

        if os.path.isfile(path):
            msg = f"Invalid input path is a file path: {path}"
            raise ProfilerPathErrorException(msg)

    @classmethod
    def check_input_file_path(cls, path: str):
        """
        Function Description:
            check whether the file path is valid, some businesses can accept a path that does not exist,
            so the function do not verify whether the path exists
        Parameter:
            path: the file path to check, whether the incoming path is absolute or relative depends on the business
        Exception Description:
            when invalid data throw exception
        """
        cls._input_path_common_check(path)

        if os.path.isdir(path):
            msg = f"Invalid input path is a directory path: {path}"
            raise ProfilerPathErrorException(msg)

        file_size = os.path.getsize(path)
        if file_size >= cls.MAX_FILE_SIZE:
            msg = f"file size exceeds the limit: {cls.MAX_FILE_SIZE}, file size: {file_size}"
            raise ProfilerPathErrorException(msg)

        file_stat = os.stat(path)
        if file_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            msg = f"File path {path} has group or others writable permissions, which is not allowed."
            raise ProfilerPathErrorException(msg)

        if stat.S_ISCHR(file_stat.st_mode) or stat.S_ISBLK(file_stat.st_mode):
            msg = f"Invalid input path is a character or block device path: {path}"
            raise ProfilerPathErrorException(msg)

    @classmethod
    def get_directory_size(cls, directory: str, unit: str = 'MB') -> float:
        """
        Function Description:
            Get the size of the directory
        Parameter:
            directory: the directory path
            unit: the unit of the size, default is MB
        Return:
            float: the size of the directory
        """
        if not os.path.exists(directory):
            logger.warning("Get directory size failed, %s not exists", directory)
            return 0.0

        cls.check_input_directory_path(directory)
        unit_map = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }

        if unit not in unit_map:
            logger.error("Invalid unit: %s", unit)
            return 0.0

        total_size = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, FileNotFoundError):
                    continue

        return total_size / unit_map[unit]

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the user
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise ProfilerPathErrorException(msg)
        if os.name != 'nt' and os.stat(path).st_uid != os.getuid():
            msg = (f"Path {path} owner[{os.stat(path).st_uid}] does not match the current user[{os.getuid()}]."
                   f"Please execute chown -R $(id -un) {path}")
            raise ProfilerPathErrorException(msg)

    @classmethod
    def check_directory_path_writeable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft link: {path}"
            raise ProfilerPathErrorException(msg)
        if not os.access(path, os.W_OK):
            msg = f"The path writeable permission check failed: {path}. Please execute chmod -R 755 {path}"
            raise ProfilerPathErrorException(msg)

    @classmethod
    def check_directory_path_readable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft link: {path}"
            raise ProfilerPathErrorException(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path readable permission check failed: {path}. Please execute chmod -R 755 {path}"
            raise ProfilerPathErrorException(msg)

    @classmethod
    def remove_path_safety(cls, path: str):
        """
        Function Description:
            remove path safety
        Parameter:
            path: the path to remove
        Exception Description:
            when invalid data throw exception
        """
        if not os.path.exists(path):
            logger.warning("The path does not exist: %s", path)
            return

        if os.path.islink(path):
            msg = f"Failed to remove path: {path}, is a soft link"
            raise ProfilerPathErrorException(msg)

        try:
            shutil.rmtree(path)
        except PermissionError as err:
            raise ProfilerPathErrorException(f"Permission denied while removing path: {path}") from err
        except Exception as err:
            raise ProfilerPathErrorException(f"Failed to remove path: {path}, err: {err}") from err

    @classmethod
    def remove_file_safety(cls, file: str):
        """
        Function Description:
            remove file safety
        Parameter:
            path: the file to remove
        Exception Description:
            when invalid data throw exception
        """
        if not os.path.exists(file):
            logger.warning("The file does not exist: %s", file)
            return

        if os.path.islink(file):
            msg = f"Failed to remove file: {file}, is a soft link"
            raise ProfilerPathErrorException(msg)

        try:
            os.remove(file)
        except PermissionError as err:
            raise ProfilerPathErrorException(f"Permission denied while removing file: {file}") from err
        except Exception as err:
            raise ProfilerPathErrorException(f"Failed to remove file: {file}, err: {err}") from err

    @classmethod
    def make_dir_safety(cls, path: str):
        """
        Function Description:
            make directory safety
        Parameter:
            path: the directory to remove
        Exception Description:
            when invalid data throw exception
        """
        if os.path.exists(path):
            return

        if os.path.islink(path):
            msg = f"Failed to make directory: {path}, is a soft link"
            raise ProfilerPathErrorException(msg)

        try:
            os.makedirs(path, mode=cls.DATA_DIR_AUTHORITY, exist_ok=True)
        except Exception as err:
            raise ProfilerPathErrorException(f"Failed to make directory: {path}, err: {err}") from err

    @classmethod
    def _input_path_common_check(cls, path: str):
        """
        Function Description:
            input path check common function
        Parameter:
            path: the file path to check
        Exception Description:
            when invalid data throw exception
        """
        if len(path) > cls.MAX_PATH_LENGTH:
            msg = f"Path {path} length {len(path)} exceeds the limit {cls.MAX_PATH_LENGTH}."
            raise ProfilerPathErrorException(msg)

        if os.path.islink(path):
            msg = f"Invalid input path is a soft link: {path}"
            raise ProfilerPathErrorException(msg)

        pattern = r"(\.|/|_|-|\s|[~0-9a-zA-Z]|[\u4e00-\u9fa5])+"
        if not re.fullmatch(pattern, path):
            msg = f"Invalid input path: {path}, contains invalid characters."
            raise ProfilerPathErrorException(msg)

        path_split_list = path.split("/")
        for name in path_split_list:
            if len(name) > cls.MAX_FILE_NAME_LENGTH:
                msg = f"Length of input path {path} file name {name} exceeds the limit {cls.MAX_FILE_NAME_LENGTH}."
                raise ProfilerPathErrorException(msg)

    @classmethod
    def get_ascend_ms_path_list(cls, input_path: str):
        """
        Function Description:
            get valid profiler {}_ascend_ms_dir path list from input_path
        Parameter:
            input_path: The directory path from which to extract profiler parent paths.
        Return:
            A list containing the input path or its subdirectories that are valid profiler parents.
        """
        if os.path.isdir(input_path) and (cls.get_fwk_path(input_path) or cls.get_cann_path(input_path)):
            return [input_path]
        sub_dirs = os.listdir(os.path.realpath(input_path))
        profiler_ascend_ms_path_list = []
        for sub_dir in sub_dirs:
            sub_path = os.path.join(input_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            if cls.get_fwk_path(sub_path) or cls.get_cann_path(sub_path):
                profiler_ascend_ms_path_list.append(os.path.join(input_path, sub_dir))
        return profiler_ascend_ms_path_list

    @classmethod
    def get_fwk_path(cls, input_path: str):
        """
        Function Description:
            get valid framework path from input_path
        Parameter:
            input_path: the directory path to check whether exist valid FRAMEWORK path
        Return:
            The path to the FRAMEWORK directory if found, otherwise an empty string.
        """
        fwk_path = os.path.join(input_path, FileConstant.FRAMEWORK_DIR)
        if os.path.isdir(fwk_path):
            return fwk_path
        return ""

    @classmethod
    def get_cann_path(cls, input_path: str):
        """
        Function Description:
            get valid PROF_XXX path from input_path
        Parameter:
            input_path: the directory path to check valid PROF_XXX path
        Return:
            The path to the PROF_XXX directory if it matches the pattern and exists, otherwise an empty string.
        """
        sub_dirs = os.listdir(os.path.realpath(input_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(input_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(FileConstant.CANN_FILE_REGEX, sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_profiler_info_path(cls, ascend_ms_dir: str) -> str:
        """
        Function Description:
            Get profiler_info_*.json path from ascend_ms_dir
        Parameter:
            ascend_ms_dir: the directory path of profiler data, eg: xxx_ascend_ms
        Return:
            str type profiler_info_*.json path
        """
        prof_info_path_pattern = os.path.join(ascend_ms_dir, "profiler_info_*.json")
        prof_info_paths = glob.glob(prof_info_path_pattern)

        if not prof_info_paths:
            raise ValueError(f"Cannot find profiler_info.json in the {ascend_ms_dir}")

        if len(prof_info_paths) > 1:
            logger.warning(
                f"There are more than one profiler_info.json in the {ascend_ms_dir}, "
                f"use the first one: {prof_info_paths[0]}"
            )
        return prof_info_paths[0]

    @classmethod
    def get_real_path(cls, path: str):
        expanded_path = os.path.expanduser(path)
        if os.path.islink(path):
            msg = f"Invalid input path is a soft link: {path}"
            raise ProfilerPathErrorException(msg)
        return os.path.realpath(expanded_path)

    @classmethod
    def check_cann_lib_valid(cls, path: str) -> bool:
        """
        Function Description:
           check if cann lib path is valid
        Parameter:
            path: the cann lib path to check
        Return:
            bool: True if the path is valid, False otherwise
        """
        lib_path = os.path.realpath(path)
        if not os.path.exists(lib_path):
            return False
        if os.path.isdir(lib_path) or os.path.islink(lib_path):
            return False
        if bool(os.stat(lib_path).st_mode & stat.S_IWOTH):
            return False
        if os.name == 'nt':
            return False
        if os.stat(lib_path).st_uid == 0 or os.stat(lib_path).st_uid == os.getuid():
            return True
        return False

    @classmethod
    def check_path_is_other_writable(cls, path):
        """Check whether the file or directory in the specified path has writable permissions for others."""
        file_stat = os.stat(path)
        if file_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            msg = (f"File path {path} has group or others writable permissions, which is not allowed."
                   f"Please execute chmod -R 755 {path}")
            raise ProfilerPathErrorException(msg)

    @classmethod
    def check_path_is_owner_or_root(cls, path):
        """Check path is owner or root."""
        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise ProfilerPathErrorException(msg)
        file_stat = os.stat(path)
        current_uid = os.getuid()
        file_uid = file_stat.st_uid
        if file_uid not in (0, current_uid):
            return False
        return True

    @classmethod
    def check_path_is_executable(cls, path):
        """Check path is executable"""
        return os.access(path, os.X_OK)

    @classmethod
    def check_path_is_readable(cls, path):
        """Check path is readable"""
        if os.path.islink(path):
            msg = f"Invalid path is a soft link: {path}"
            raise ProfilerPathErrorException(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path readable permission check failed: {path}."
            raise ProfilerPathErrorException(msg)

    @classmethod
    def walk_with_depth(cls, path, *args, max_depth=10, **kwargs):
        """walk path depth"""
        if not isinstance(path, str):
            return
        base_depth = path.count(os.sep)
        if path.endswith(os.sep):
            base_depth -= 1
        for root, dirs, files in os.walk(path, *args, **kwargs):
            if root.count(os.sep) - base_depth > max_depth:
                dirs.clear()
                continue
            yield root, dirs, files
