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

"""This Model use to process compile json to mapping json item."""
import os
import sys

key = '\"file\":'


def extract_and_process_lines(root_path):
    """
    Transpose compile json to mapping json
    """
    for filename in os.listdir(root_path):
        unique_directories = set()
        if filename.startswith('Compile') and filename.endswith('.json'):
            file_path = os.path.join(root_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    if key in line:
                        parts = line.split('/')
                        path_pre = '/'.join(parts[:-1])
                        unique_directories.add(path_pre)

        sorted_unique_lines = list(unique_directories)
        sorted_unique_lines.sort()
        output_file_path = os.path.join(root_path, 'processed_' + filename)
        print("proce file name:", output_file_path)

        final_lines = []
        for i, line in enumerate(sorted_unique_lines):
            stripped_line = line.strip()
            is_subset = False
            for j, other_line in enumerate(sorted_unique_lines):
                other_stripped_line = other_line.strip()
                if i != j and stripped_line != other_stripped_line and other_stripped_line in stripped_line:
                    is_subset = True
                    break
            if not is_subset:
                if 'akg/' in stripped_line or 'third_party/' in stripped_line or 'build/' in stripped_line:
                    continue
                start_index = stripped_line.find('mindspore')
                if start_index != -1:
                    new_line = stripped_line[start_index + len('mindspore') + 1:]
                    new_line = '"' + new_line + '",\n'
                    final_lines.append(new_line)

        if not final_lines:
            with open(output_file_path, 'w') as file:
                file.writelines(final_lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        sys.exit(1)

    extract_and_process_lines(directory_path)
