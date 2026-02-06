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
import os
import re
import csv
import json


class FileChecker:
    @classmethod
    def check_file_exists(cls, file_path: str) -> None:
        """
        Check if a file whether exist.

        Args:
            file_path (str): file path to check.
        """
        assert os.path.exists(file_path) and os.path.isfile(file_path), f"The {file_path} does not exist"

    @classmethod
    def check_txt_not_empty(cls, txt_path: str) -> None:
        """
        Check if a TXT file is not empty.

        Args:
            txt_path (str): Path to the TXT file.
        """
        try:
            cls.check_file_exists(txt_path)
            with open(txt_path, 'r', encoding='utf-8') as txtfile:
                content = txtfile.read()
                assert bool(content), f"The file {txt_path} is empty."
        except (IOError, OSError) as e:
            assert False, f"Failed to read TXT file, ERROR: {e}"

    @classmethod
    def check_csv_headers(cls, csv_path: str, headers: list) -> None:
        """
        Check if the headers of a CSV file match the given headers list.

        Args:
            csv_path (str): Path to the CSV file.
            headers (list): List of expected headers.

        Example:
            To verify that a CSV file contains the headers "Op Name" and "Op Type", you can call the method like this:
            FileChecker.check_csv_headers(csv_path, ["Op Name", "Op Type"])
        """
        try:
            cls.check_file_exists(csv_path)
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                first_row = next(reader)  # Get the first row
                csv_headers_set = set(first_row)
                expected_headers_set = set(headers)
                assert expected_headers_set.issubset(csv_headers_set), (f"Missing headers: "
                                                                        f"{expected_headers_set - csv_headers_set}")
        except (IOError, OSError) as e:
            assert False, f"Failed to read CSV file, ERROR: {e}"

    @classmethod
    def check_csv_items(cls, csv_path: str, item_pattern: dict, fuzzy_match: bool = True) -> None:
        """
        Check if items in specified columns of a CSV file match given patterns, including fuzzy match

        Args:
            csv_path (str): Path to the CSV file.
            item_pattern (dict): Dictionary containing patterns.
            fuzzy_match (bool, optional): Whether to enable fuzzy matching using regex. Defaults to True.

        Example:
            Given a CSV file with the following content:
            Op Name                        Op Type
            aclnnAdd_AddAiCore_Add         Add
            aclnnMul_MulAiCore_Mul         Mul

            Use the following call to match patterns:
            FileChecker.check_csv_items(csv_path, {"Op Name": ["*Add*", "*Mul*"], "Op Type": "Add"}, fuzzy_match=True)
            This will match because "Op Name" contains "Add" and "Mul", and "Op Type" contains "Add"
        """
        try:
            cls.check_file_exists(csv_path)
            cls.check_csv_headers(csv_path, list(item_pattern.keys()))
            reader = csv.DictReader(open(csv_path, 'r', newline='', encoding='utf-8'))
            csv_data = list(reader)
            for column, patterns in item_pattern.items():
                patterns = [patterns] if not isinstance(patterns, list) else patterns
                if fuzzy_match:
                    regex_patterns = [re.compile(re.escape(pattern).replace(r'\*', '.*'), re.IGNORECASE)
                                      for pattern in patterns]
                    found_match = all(any(rp.search(row[column]) for row in csv_data) for rp in regex_patterns)
                else:
                    found_match = all(any(row[column] == pattern for row in csv_data) for pattern in patterns)
                assert found_match, f"No value in column '{column}' matches patterns '{patterns}'"
        except (IOError, OSError) as e:
            assert False, f"Failed to read CSV file, ERROR: {e}"

    @classmethod
    def check_timeline_values(cls, timeline_path: str, key: str = "name", value_list: list = None,
                              fuzzy_match: bool = True) -> None:
        """
        Check if a timeline file contains the specified list of values for a given key.

        Args:
            timeline_path (str): Path to the JSON file.
            key (str, optional): The key to check in the timeline data. Defaults to "name".
            value_list (list, optional): List of values to check for the specified key. Defaults to None.
            fuzzy_match (bool, optional): Whether to enable fuzzy matching. Defaults to True.

        Example:
            Given a timeline JSON file with the following content:
            [
                {"name": "event1", "duration": 100},
                {"name": "event2", "duration": 200},
                {"name": "event3", "duration": 300}
            ]
            To check if the timeline contains events with names like "event*":
            FileChecker.check_timeline_values(timeline_path, "name", key=["event*"], fuzzy_match=True)
            It will verify that events with names starting with "event" exist in the "name" field of the timeline file.
        """
        if not value_list:
            value_list = []
        try:
            cls.check_file_exists(timeline_path)
            with open(timeline_path, 'r', encoding='utf-8') as timelinefile:
                data = json.load(timelinefile)
                for value in value_list:
                    if fuzzy_match:
                        pattern = re.compile(re.escape(value).replace(r'\*', '.*'), re.IGNORECASE)
                        found_match = any(pattern.search(item.get(key, "")) for item in data)
                    else:
                        found_match = any(item.get(key, None) == value for item in data)
                    assert found_match, f"Value '{value}' for key '{key}' not found in Timeline file."
        except (IOError, OSError, json.JSONDecodeError) as e:
            assert False, f"Failed to read Timeline file, ERROR: {e}"

    @classmethod
    def check_json_items(cls, json_path: str, item_pattern: dict) -> None:
        """
        Check if a JSON file contains the specified keys with the expected values, including nested keys.

        Args:
            json_path (str): Path to the JSON file.
            item_pattern (dict): Dictionary containing expected key-value pairs, including nested keys.

        Example:
            Given a JSON file with the following content:
            {
                "a": {
                    "b": {
                        "c": "value"
                    }
                }
            }
            You can set item_pattern to {"a.b.c": "value"} and call:
            FileChecker.check_json_items(json_path, {"a.b.c": "value"})
            This will verify that the value associated with the nested key "a.b.c" is "value".
        """
        try:
            cls.check_file_exists(json_path)
            with open(json_path, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                for nested_key, value in item_pattern.items():
                    keys = nested_key.split('.')
                    current = data
                    for key in keys:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            assert False, f"Key '{nested_key}' not found in JSON file."
                    assert current == value, (f"Value for key '{nested_key}' does not match. "
                                              f"Expected '{value}', found '{current}'")
        except (IOError, OSError, json.JSONDecodeError) as e:
            assert False, f"Failed to read JSON file, ERROR: {e}"

    @classmethod
    def check_json_keys(cls, json_path: str, keys: list) -> None:
        """
        Check if a JSON file contains the specified keys with the expected values, including nested keys.

        Args:
            json_path (str): Path to the JSON file.
            keys (list): Dictionary containing expected keys

        Example:
            Given a JSON file with the following content:
            {
                "a": "1",
                "b": "2"
            }
            You can set keys to ["a", "b"] and call:
            FileChecker.check_json_keys(json_path, ["a", "b"])
        """
        try:
            cls.check_file_exists(json_path)
            with open(json_path, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                for key in keys:
                    assert key in data, f"Key '{key}' not found in JSON file."
        except (IOError, OSError, json.JSONDecodeError) as e:
            assert False, f"Failed to read JSON file, ERROR: {e}"

    @classmethod
    def check_file_line_count(cls, file_path: str, expected_line_count: int) -> None:
        """
        Check if a file (CSV or TXT) contains the expected number of lines.

        Args:
            file_path (str): Path to the file.
            expected_line_count (int): The expected number of lines in the file.
        """
        try:
            cls.check_file_exists(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            assert (line_count == expected_line_count), (f"Expected {expected_line_count} lines, "
                                                         f"but found {line_count} in file {file_path}.")
        except (IOError, OSError) as e:
            assert False, f"Failed to read file, ERROR: {e}"

    @classmethod
    def assert_csv_no_header(cls, csv_path: str, header_to_check: str) -> None:
        """
        Check if the headers of a CSV file not contain the given headers list.

        Args:
        csv_path (str): Path to the CSV file.
        header_to_check (str): The header to check for absence.

        Raises:
        AssertionError: If the header is found in the CSV file.
        """
        try:
            cls.check_file_exists(csv_path)
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Read the first row which contains headers
                assert header_to_check not in headers, (f"The column '{header_to_check}' should not be present in the "
                                                        f"CSV file.")
        except (IOError, OSError) as e:
            assert False, f"Failed to read CSV file or file is empty, ERROR: {e}"

    @classmethod
    def check_file_for_keyword(cls, file_path: str, keyword: str) -> None:
        """
        Check if a file contains a specific keyword (case-insensitive).

        Args:
            file_path (str): Path to the file.
            keyword (str): The keyword to search for in the file.
        """
        try:
            cls.check_file_exists(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            if keyword.lower() in content.lower():  # Convert both content and keywords to lowercase and then check
                assert False, f"file {file_path} contains the keyword '{keyword}'."
        except (IOError, OSError) as e:
            assert False, f"Failed to read file, ERROR: {e}"

    @classmethod
    def check_csv_data_non_negative(cls, file_path: str, comparison_func) -> None:
        """
        Check if all data in the CSV file (except the header row) are greater than or equal to the specified threshold.

        Args:
            file_path (str): Path to the CSV file.
            comparison_func (callable): A function that takes a value and returns a boolean.
        """
        try:
            cls.check_file_exists(file_path)

            with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                # pylint: disable=unused-variable
                headers = next(reader)  # Skip head

                for row in reader:
                    for value in row:
                        # Check whether the conditions are met
                        if not comparison_func(value):
                            raise AssertionError(f"CSV file contains values that do not meet the condition: {value}.")
        except (IOError, OSError) as e:
            assert False, f"Failed to read CSV file, ERROR: {e}"
