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
import yaml


py_licence_str = f"""# Copyright 2023 Huawei Technologies Co., Ltd
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

cc_license_str = f"""/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */"""


def is_tensor(op_arg):
    if op_arg.arg_dtype == 'tensor':
        return True
    return False


def is_tensor_list(op_arg):
    if op_arg.arg_dtype in ['list[tensor]', 'tuple[tensor]']:
        return True
    return False


def is_list(op_arg):
    if op_arg.arg_dtype in ['tuple[int]', 'tuple[float]', 'tuple[bool]',
                            'tuple[tensor]', 'list[int]', 'list[bool]', 'list[tensor]']:
        return True
    return False


def get_index(index: int):
    """
    get index
    :param index:
    :return: str
    """
    index_map = {
        0: 'kIndex0',
        1: 'kIndex1',
        2: 'kIndex2',
        3: 'kIndex3',
        4: 'kIndex4',
        5: 'kIndex5',
        6: 'kIndex6',
        7: 'kIndex7',
        8: 'kIndex8',
        9: 'kIndex9',
    }
    if index in index_map:
        return index_map[index]
    raise TypeError(f"""Unsupported index {index} for index map.""")


def get_convert_type_str(dtype: str):
    """
    Convert type
    """
    # add more type here
    type_convert = {
        'int': 'ToInt',
        'float': 'ToFloat',
        'bool': 'ToBool',
        'number': 'ToScalar',
        'tuple[int]': 'ToIntList<py::tuple>',
        'tuple[float]': 'ToFloatList<py::tuple>',
        'tuple[bool]': 'ToBoolList<py::tuple>',
        'tuple[tensor]': 'ToTensorList<py::tuple>',
        'list[int]': 'ToIntList<py::list>',
        'list[float]': 'ToFloatList<py::list>',
        'list[bool]': 'ToBoolList<py::list>',
        'list[tensor]': 'ToTensorList<py::list>',
        'tensor': 'ToTensor',
        'type': 'ToDtype',
    }
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")

def tuple_input_to_cpp_type(dtype: str):
    types_map = {
        'tuple[int]': 'int64_t',
        'tuple[float]': 'double',
        'tuple[bool]': 'bool',
        'list[int]': 'int64_t',
        'list[float]': 'double',
        'list[bool]': 'bool',
    }
    if dtype in types_map:
        return types_map[dtype]
    return None

def number_input_to_cpp_type(dtype: str):
    types_map = {
        'int': 'int64_t',
        'float': 'double',
        'bool': 'bool',
    }
    if dtype in types_map:
        return types_map[dtype]
    return None

def get_input_dtype(dtype: str):
    """
    Convert type
    """
    # add more type here
    type_convert = {
        'int': 'Int64ImmPtr',
        'float': 'FP64ImmPtr',
        'bool': 'BoolImmPtr',
        'number': 'ScalarPtr',
        'tuple[int]': 'ValueTuplePtr',
        'tuple[float]': 'ValueTuplePtr',
        'tuple[bool]': 'ValueTuplePtr',
        'tuple[tensor]': 'ValueTuplePtr',
        'list[int]': 'ValueTuplePtr',
        'list[float]': 'ValueTuplePtr',
        'list[bool]': 'ValueTuplePtr',
        'list[tensor]': 'ValueTuplePtr',
        'tensor': 'TensorPtr',
        'type': 'TypePtr',
    }
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")

def get_return_type(dtype: str):
    """
    Convert type
    """
    # add more type here
    type_convert = {
        'tuple[tensor]': 'std::vector<TensorPtr>',
        'list[tensor]': 'std::vector<TensorPtr>',
        'tensor': 'TensorPtr',
    }
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")

def get_type_str(type_str):
    """
    Get the unified type str for operator arg dtype.
    """
    # add more type here
    type_kind_dict = {
        'int': 'OpDtype.PY_DT_INT',
        'float': 'OpDtype.PY_DT_FLOAT',
        'bool': 'OpDtype.PY_DT_BOOL',
        'number': 'OpDtype.PY_DT_NUMBER',
        'tuple[int]': 'OpDtype.PY_DT_TUPLE_ANY',
        'tuple[float]': 'OpDtype.PY_DT_TUPLE_ANY',
        'tuple[bool]': 'OpDtype.PY_DT_TUPLE_ANY',
        'tuple[tensor]': 'OpDtype.PY_DT_TUPLE_ANY',
        'list[int]': 'OpDtype.PY_DT_LIST_ANY',
        'list[float]': 'OpDtype.PY_DT_LIST_ANY',
        'list[bool]': 'OpDtype.PY_DT_LIST_ANY',
        'list[tensor]': 'OpDtype.PY_DT_LIST_ANY',
        'tensor': 'OpDtype.PY_DT_TENSOR',
        'type': 'OpDtype.PY_DT_TYPE',
    }
    if type_str in type_kind_dict:
        return type_kind_dict[type_str]
    raise TypeError(f"""Unsupported type {type_str} for args.""")


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
        with open(file_path, 'r') as file:
            merged_content += file.read()
            merged_content += '\n'
    one_file = open(merged_file_path, 'w')
    one_file.write(merged_content)
    one_file.close()


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
    yaml_str = dict()
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_str.update(yaml.safe_load(yaml_file))
    return yaml_str


def get_assign_str_by_type_it(arg_info, arg_name, dtype):
    """
    Make type_it(arg, src_types, dst_type) python sentences.
    """
    assign_str = ""
    type_cast = arg_info.get('type_cast')
    if type_cast is not None:
        type_cast_tuple = tuple(ct.strip() for ct in type_cast.split(","))
        assign_str += f'type_it({arg_name}, '
        if len(type_cast_tuple) == 1:
            assign_str += get_type_str(type_cast_tuple[0]) + '.value, '
        else:
            assign_str += '(' + ', '.join(get_type_str(ct) + '.value' for ct in type_cast_tuple) + '), '
        assign_str += get_type_str(dtype) + '.value)'
    else:
        assign_str = arg_name
    return assign_str


def convert_python_func_name_to_c(func_name: str) -> str:
    return ''.join(word.capitalize() for word in func_name.split('_'))
