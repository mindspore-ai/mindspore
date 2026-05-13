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
"""Profiler file manager"""
import csv
import json
import os
import shutil
from typing import List, Dict, Optional, Tuple
import numpy as np

from mindspore import log as logger
from mindspore.profiler.common.path_manager import PathManager


class FileManager:
    """Profiler file manager"""

    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o700
    FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

    @classmethod
    def read_file_content(cls, path: str, mode: str = "r"):
        """Read the content in the input file."""
        PathManager.check_input_file_path(path)
        PathManager.check_directory_path_readable(path)
        try:
            with open(path, mode) as file:  # pylint: disable=W1514
                return file.read()
        except Exception as err:
            raise RuntimeError(f"Failed read file: {path}, error: {err}") from err

    @classmethod
    def read_json_file(cls, file_path: str) -> Optional[Dict]:
        """Read json file and return dict data"""
        PathManager.check_input_file_path(file_path)
        PathManager.check_directory_path_readable(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                return data
        except Exception as err:
            raise RuntimeError(
                f"Failed read json file: {file_path}, error: {err}"
            ) from err

    @classmethod
    def create_json_file(
            cls, output_file_path: str, json_data: List, indent: int = None
    ) -> None:
        """Create json file with least authority"""
        PathManager.check_directory_path_writeable(os.path.dirname(output_file_path))
        if not json_data:
            logger.warning("Json data is empty, file path: %s", output_file_path)
            return

        try:
            with os.fdopen(
                    os.open(output_file_path, cls.FLAGS, cls.DATA_FILE_AUTHORITY), "w"
            ) as fp:
                data = json.dumps(json_data, indent=indent, ensure_ascii=False)
                fp.write(data)
        except Exception as err:
            raise RuntimeError(
                f"Failed create json file: {output_file_path}, error: {err}"
            ) from err

    @classmethod
    def read_csv_file(cls, file_path: str) -> list:
        """Read csv file and return list"""
        PathManager.check_input_file_path(file_path)
        PathManager.check_directory_path_readable(file_path)
        result_data = []
        try:
            with open(file_path, newline="", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    result_data.append(row)
            return result_data
        except Exception as err:
            raise RuntimeError(f"Failed read csv file: {file_path}, error: {err}") from err

    @classmethod
    def read_csv_file_as_numpy(
            cls, file_path: str, extern_headers: list = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Read csv file and return numpy array"""
        PathManager.check_input_file_path(file_path)
        PathManager.check_directory_path_readable(file_path)
        try:
            with open(file_path, newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
                headers = reader.fieldnames
                if extern_headers:
                    headers = extern_headers + headers

                csv_data = [
                    tuple(row.get(field) for field in headers) for row in reader
                ]
                csv_data_np = np.array(
                    csv_data, dtype=np.dtype([(field, object) for field in headers])
                )

            return csv_data_np, headers
        except Exception as err:
            raise RuntimeError(f"Failed read csv file: {file_path}, error: {err}") from err

    @classmethod
    def create_csv_file(cls, file_path: str, data: list, headers: list = None) -> None:
        """Create csv file and write the data"""
        if not data:
            logger.error(
                "Create csv file failed, data is empty, file path: %s", file_path
            )
            return
        PathManager.check_directory_path_writeable(os.path.dirname(file_path))
        try:
            with os.fdopen(
                    os.open(file_path, cls.FLAGS, cls.DATA_FILE_AUTHORITY), "w"
            ) as fp:
                writer = csv.writer(fp)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
        except Exception as err:
            raise RuntimeError(
                f"Failed create csv file: {file_path}, error: {err}"
            ) from err

    @classmethod
    def combine_csv_file(
            cls, source_file_list: list, target_file_path: str, header_map: dict = None
    ):
        """Merge multiple CSV files into one"""
        headers, all_data = [], []
        for source_file in source_file_list:
            data = cls.read_csv_file(source_file)
            if len(data) > 1:
                headers = data[0]
                all_data.extend(data[1:])
        if all_data:
            if isinstance(header_map, dict):
                headers = [header_map.get(header, header) for header in headers]
            FileManager.create_csv_file(target_file_path, all_data, headers)

    @classmethod
    def read_txt_file(cls, file_path: str) -> list:
        """Read txt file and return list"""
        PathManager.check_input_file_path(file_path)
        PathManager.check_directory_path_readable(file_path)
        result_data = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file.readlines():
                    result_data.append(line.strip())
        except Exception as err:
            raise RuntimeError(f"Failed read txt file: {file_path}, error: {err}") from err
        return result_data

    @classmethod
    def copy_file(cls, src_path: str, dst_path: str):
        """
        Function Description:
            copy file safety
        Parameter:
            src_path: file source path
            dst_path: file destination path
        Exception Description:
            when src_path is link throw exception
        """
        if not os.path.exists(src_path):
            logger.warning("The source file does not exist: %s", src_path)
            return

        PathManager.check_input_file_path(src_path)
        src_dir = os.path.dirname(src_path)
        PathManager.check_directory_path_readable(src_dir)
        dst_dir = os.path.dirname(dst_path)
        PathManager.check_directory_path_writeable(dst_dir)

        try:
            shutil.copy2(src_path, dst_path)
        except (shutil.Error, IOError) as err:
            msg = f"Failed to copy from '{src_path}' to '{dst_path}': {err}"
            raise RuntimeError(msg) from err

    @classmethod
    def get_csv_file_list_by_start_name(cls, source_path: str, start_name: str):
        """Get all the csv files that match the name"""
        file_list = []
        for file_name in os.listdir(source_path):
            if file_name.startswith(start_name) and file_name.endswith(".csv"):
                file_list.append(os.path.join(source_path, file_name))
        return file_list

    @classmethod
    def check_file_owner(cls, path):
        """Check whether the file owner is the current user or root."""
        stat_info = os.stat(path)
        if stat_info.st_uid == 0:
            return True
        current_uid = os.geteuid()
        return current_uid == stat_info.st_uid
