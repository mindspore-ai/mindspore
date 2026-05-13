# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"""
Generate operator utils function
"""
import os
import glob
import hashlib
import pathlib
import re
import logging
import json
from collections import OrderedDict

import yaml

from . import gen_constants as K


def convert_dtype_str(dtype_str):
    """
    Convert dtype str to expression in ops file
    """
    return 'DT_' + dtype_str.replace('[', '_').replace(']', '').upper()


def get_type_str(type_str):
    """
    Get the unified type str for operator arg dtype.
    """
    # add more type here
    type_kind_set = {
        'int',
        'float',
        'bool',
        'number',
        'tuple[int]',
        'tuple[float]',
        'tuple[bool]',
        'tuple[tensor]',
        'tuple[str]',
        'list[int]',
        'list[float]',
        'list[bool]',
        'list[tensor]',
        'list[str]',
        'tensor',
        'type',
    }
    if type_str in type_kind_set:
        return "OpDtype." + convert_dtype_str(type_str)
    raise TypeError("""Unsupported type {type_str} for args.""")


def get_file_md5(file_path):
    """
    Get the md5 value for file.
    """
    if not os.path.exists(file_path):
        return ""
    if os.path.isdir(file_path):
        return ""
    with open(file_path, 'rb') as f:
        data = f.read()
    md5_value = hashlib.md5(data).hexdigest()
    return md5_value


def check_change_and_replace_file(last_file_path, tmp_file_path):
    """
    Compare tmp_file with the md5 value of the last generated file.
    If the md5 value is the same, retain the last generated file.
    Otherwise, update the last generated file to tmp_file.
    """
    last_md5 = get_file_md5(last_file_path)
    tmp_md5 = get_file_md5(tmp_file_path)

    if last_md5 == tmp_md5:
        os.remove(tmp_file_path)
    else:
        if os.path.exists(last_file_path):
            os.remove(last_file_path)
        os.rename(tmp_file_path, last_file_path)


def merge_files_to_one_file(file_paths, merged_file_path):
    """
    Merge multiple files into one file.
    """
    merged_content = ''
    file_paths.sort()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            merged_content += file.read()
            merged_content += '\n'
    with open(merged_file_path, 'w', encoding='utf-8') as file:
        file.write(merged_content)


def merge_files(origin_dir, merged_file_path, file_format):
    """
    Merge multiple files into one file.
    origin_dir: indicates the origin file directory.
    merged_file_path: indicates the merged file path.
    file_format: indicates the format of regular matching.
    Files whose names meet the regular matching in 'origin_dir' directory will be merged into one file.
    """
    op_yaml_file_names = glob.glob(os.path.join(origin_dir, file_format))
    merge_files_to_one_file(op_yaml_file_names, merged_file_path)


def safe_load_yaml(yaml_file_path):
    """
    Load yaml dictionary from file.
    """
    if not yaml_file_path.endswith('.yaml'):
        raise ValueError(f"Invalid yaml file path: {yaml_file_path}")

    with open(yaml_file_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)


def safe_load_yaml_from_dir(yaml_dir_path):
    """
    Load yaml dictionary from directory.
    """
    yaml_dict = {}
    if not os.path.isdir(yaml_dir_path):
        raise ValueError(f"Invalid yaml directory path: {yaml_dir_path}")

    for yaml_file_path in glob.glob(os.path.join(yaml_dir_path, '*.yaml')):
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        for name, data in yaml_data.items():
            if name in yaml_dict:
                raise ValueError(f"Duplicate yaml object with name '{name}'.")
            yaml_dict[name] = data

    return yaml_dict


def get_assign_str_by_type_it(class_name, arg_info, arg_name, dtype):
    """
    Make type_it(arg, src_types, dst_type) python sentences.
    """
    assign_str = ""
    type_cast = arg_info.get('type_cast')
    if type_cast is not None:
        type_cast_tuple = tuple(ct.strip() for ct in type_cast.split(","))
        assign_str += f"type_it('{class_name}', '{arg_name}', {arg_name}, "
        if len(type_cast_tuple) == 1:
            assign_str += get_type_str(type_cast_tuple[0]) + ', '
        else:
            assign_str += '(' + ', '.join(get_type_str(ct) for ct in type_cast_tuple) + '), '
        assign_str += get_type_str(dtype) + ')'
    else:
        assign_str = arg_name
    return assign_str


gen_file_lists = set()  # list of all generated files


def save_file(save_path, file_name, content):
    """
    Save file to path. For existing file, only update the file if the content is different.
    """
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    dst_file_path = os.path.join(save_path, file_name)
    gen_file_lists.add(os.path.abspath(dst_file_path))

    if os.path.exists(dst_file_path):
        if get_file_md5(dst_file_path) == hashlib.md5(content.encode('utf-8')).hexdigest():
            return

    with open(dst_file_path, "w", encoding='utf-8') as f:
        f.write(content)


def clear_obsolete_auto_gen_files():
    """
    Clear obsolete auto-generated files.
    """
    gen_file_list_filename = os.path.join(K.WORK_DIR, "gen_file_lists.json")
    if not os.path.exists(gen_file_list_filename):
        last_gen_file_lists = set()
    else:
        with open(gen_file_list_filename, "r", encoding='utf-8') as f:
            last_gen_file_lists = set(json.load(f))

    obsolete_files = last_gen_file_lists - gen_file_lists
    for file in obsolete_files:
        if os.path.isfile(file):
            os.remove(file)

    with open(gen_file_list_filename, "w", encoding='utf-8') as f:
        json.dump(list(gen_file_lists), f)


def normalize_func_description_format(description):
    """
    Process description.
    """
    if not description:
        return description
    lines = description.split("\n")
    if len(lines) == 1:
        return description

    # Add line indentation to other lines after the first line
    for i in range(1, len(lines)):
        indent = "    " if lines[i] else ""
        lines[i] = indent + lines[i]

    # Remove trailing blank lines
    lines = lines if lines[-1] != "" else lines[:-1]
    description = "\n".join(lines)
    return description


def get_op_description(operator_name, doc_dict):
    """
    Generate ops api description.
    """
    op_description = "    r\"\"\"\n" \
                     "    \n" \
                     "    \"\"\"\n"
    if doc_dict is None:
        logging.info("Description is None, op_name: %s", operator_name)
        return op_description

    description = doc_dict.get(operator_name)
    if description is None:
        logging.info("Description is None, op_name: %s", operator_name)
        return op_description

    description = description.get("description")
    if description is None:
        logging.info("Description is None, op_name: %s", operator_name)
        return op_description

    op_description = "    r\"\"\"\n" \
                     f"    {normalize_func_description_format(description)}\n" \
                     "    \"\"\"\n"
    return op_description


def get_same_dtype_groups(args_signature, args_name):
    """
    Get same dtype groups
    """
    same_dtype_groups = {}
    dtype_count = 0

    if not args_signature:
        return same_dtype_groups, dtype_count

    dtype_group = args_signature.dtype_group
    if not args_signature.dtype_group:
        return same_dtype_groups, dtype_count

    args_list = []
    match = re.findall(r'\((.*?)\)', dtype_group)
    for item in match:
        args_list.append(item.replace(' ', '').split(","))
    for arg_name in args_name:
        if arg_name in same_dtype_groups:
            continue
        is_match = False
        for group in args_list:
            if arg_name in group:
                is_match = True
                for item in group:
                    same_dtype_groups[item] = dtype_count
                break
        if not is_match:
            same_dtype_groups[arg_name] = dtype_count
        dtype_count = dtype_count + 1
    return same_dtype_groups, dtype_count


def init_args_signature_rw(args_signature):
    """
    Extracts read, write, and reference argument lists from signature data.

    Args:
        args_signature (object): Contains 'rw_write', 'rw_read', 'rw_ref' attributes as comma-separated strings.

    Returns:
        tuple: Lists of read-only, reference, and write-only argument names.
    """
    write_list = []
    read_list = []
    ref_list = []
    if args_signature:
        if args_signature.rw_write:
            write_list.extend(args_signature.rw_write.replace(' ', '').split(","))

        if args_signature.rw_read:
            read_list.extend(args_signature.rw_read.replace(' ', '').split(","))

        if args_signature.rw_ref:
            ref_list.extend(args_signature.rw_ref.replace(' ', '').split(","))

    return read_list, ref_list, write_list


class OrderedSet:
    def __init__(self):
        self._dict = OrderedDict()

    def add(self, item):
        self._dict[item] = None

    def __iter__(self):
        return iter(self._dict.keys())
