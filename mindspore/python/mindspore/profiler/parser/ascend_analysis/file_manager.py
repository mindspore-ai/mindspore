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
import os.path
from typing import List

from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.constant import Constant


class FileManager:
    """Profiler file manager"""

    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o700

    @classmethod
    def read_file_content(cls, path: str, mode: str = "r"):
        """Read the content in the input file."""
        if not os.access(path, os.R_OK):
            msg = f"The file {os.path.basename(path)} is not readable!"
            raise RuntimeError(msg)

        if not os.path.isfile(path):
            raise RuntimeError(f"The file {os.path.basename(path)} is invalid!")
        file_size = os.path.getsize(path)
        if file_size <= 0:
            return ""
        if file_size > Constant.MAX_FILE_SIZE:
            msg = f"File too large file to read: {path}"
            logger.warning(msg)
            return ''
        try:
            with open(path, mode) as file:
                return file.read()
        except Exception as err:
            raise RuntimeError(f"Failed to read file: {path}") from err

    @classmethod
    def make_dir_safety(cls, dir_path: str):
        """Make directory with least authority"""
        dir_path = validate_and_normalize_path(dir_path)

        if os.path.exists(dir_path):
            return
        try:
            os.makedirs(dir_path, mode=cls.DATA_DIR_AUTHORITY, exist_ok=True)
        except Exception as err:
            msg = f"Failed to make directory: {dir_path}"
            raise RuntimeError(msg) from err

    @classmethod
    def create_json_file(cls, output_path: str, json_data: List, file_name: str) -> None:
        """Create json file with least authority"""
        if not json_data:
            return
        cls.make_dir_safety(output_path)
        file_path = os.path.join(output_path, file_name)
        flags = os.O_WRONLY | os.O_CREAT
        with os.fdopen(os.open(file_path, flags, cls.DATA_FILE_AUTHORITY), 'w') as fp:
            json.dump(json_data, fp, ensure_ascii=False)

    @classmethod
    def read_csv_file(cls, file_path: str) -> list:
        """Read csv file and return list"""
        if not os.path.isfile(file_path):
            return []
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > Constant.MAX_CSV_SIZE:
            msg = f"The file size exceeds the preset value, please check the file: {file_path}"
            logger.warning(msg)
            return []
        result_data = []
        try:
            with open(file_path, newline="") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    result_data.append(row)
        except Exception as err:
            raise RuntimeError(f"Failed to read the file: {file_path}") from err
        return result_data

    @classmethod
    def create_csv_file(cls, file_path: str, data: list, headers: list = None) -> None:
        """Create csv file and write the data"""
        if not data:
            return
        try:
            with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, 0o600),
                           'w', newline="") as file:
                writer = csv.writer(file)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
        except Exception as err:
            raise RuntimeError(f"Can't create file: {file_path}") from err

    @classmethod
    def combine_csv_file(cls, source_file_list: list, target_file_path: str, header_map: dict = None):
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
    def get_csv_file_list_by_start_name(cls, source_path: str, start_name: str):
        """Get all the csv files that match the name"""
        file_list = []
        for file_name in os.listdir(source_path):
            if file_name.startswith(start_name) and file_name.endswith(".csv"):
                file_list.append(os.path.join(source_path, file_name))
        return file_list

    @classmethod
    def read_txt_file(cls, file_path: str) -> list:
        """Read txt file and return list"""
        if not os.path.isfile(file_path):
            return []
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > Constant.MAX_FILE_SIZE:
            msg = f"The file size exceeds the preset value, please check the file: {file_path}"
            logger.warning(msg)
            return []
        result_data = []
        try:
            with open(file_path, "r") as file:
                for line in file.readlines():
                    result_data.append(line.strip().split(","))
        except Exception as err:
            raise RuntimeError(f"Failed to read the file: {file_path}") from err
        return result_data
