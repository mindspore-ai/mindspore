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
"""pyboost utils."""

import os
from gen_utils import safe_load_yaml


def is_optional_param(op_arg):
    if op_arg.as_init_arg and str(op_arg.default) == 'None':
        return True
    return False


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
        10: 'kIndex10',
    }
    if index in index_map:
        return index_map[index]
    raise TypeError(f"""Unsupported index {index} for index map.""")


def get_convert_type_str(dtype: str, optional):
    """
    Convert type
    """
    # add more type here
    native_type_convert = {
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
        'str': 'ToString',
        'type': 'ToDtype',
    }
    optional_type_convert = {
        'int': 'ToIntOptional',
        'float': 'ToFloatOptional',
        'number': 'ToScalarOptional',
        'tensor': 'ToTensorOptional',
        'tuple[int]': 'ToIntListOptional<py::tuple>',
        'tuple[float]': 'ToFloatListOptional<py::tuple>',
        'tuple[bool]': 'ToBoolListOptional<py::tuple>',
        'list[int]': 'ToIntListOptional<py::list>',
        'list[float]': 'ToFloatListOptional<py::list>',
        'list[bool]': 'ToBoolListOptional<py::list>',
    }
    if optional:
        if dtype in optional_type_convert:
            return optional_type_convert[dtype]
        raise TypeError(f"""Unsupported convert optional type {dtype} for args.""")
    if dtype in native_type_convert:
        return native_type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")


def tuple_input_to_cpp_type(dtype: str):
    """
    dtype convert
    :param dtype:
    :return:
    """
    types_map = {
        'tuple[int]': 'int64_t',
        'tuple[float]': 'float',
        'tuple[bool]': 'bool',
        'tuple[str]': 'string',
        'tuple[tensor]': 'TensorPtr',
        'list[int]': 'int64_t',
        'list[float]': 'float',
        'list[bool]': 'bool',
        'list[tensor]': 'TensorPtr',
    }
    return types_map.get(dtype)


def number_input_to_cpp_type(dtype: str):
    types_map = {
        'int': 'int64_t',
        'float': 'float',
        'bool': 'bool',
        'str': 'string'
    }
    return types_map.get(dtype)


def get_input_dtype(dtype: str, optional):
    """
    Convert type
    """
    # add more type here
    kValueTuplePtr = 'ValueTuplePtr'
    type_convert = {
        'int': 'Int64ImmPtr',
        'float': 'FP32ImmPtr',
        'bool': 'BoolImmPtr',
        'number': 'ScalarPtr',
        'tuple[int]': kValueTuplePtr,
        'tuple[float]': kValueTuplePtr,
        'tuple[bool]': kValueTuplePtr,
        'tuple[tensor]': kValueTuplePtr,
        'list[int]': kValueTuplePtr,
        'list[float]': kValueTuplePtr,
        'list[bool]': kValueTuplePtr,
        'list[tensor]': kValueTuplePtr,
        'tensor': 'TensorPtr',
        'str': 'StringImmPtr',
        'type': 'TypePtr',
    }
    kValueTuplePtrOptional = 'std::optional<ValueTuplePtr>'
    optional_type_convert = {
        'int': 'std::optional<Int64ImmPtr>',
        'float': 'std::optional<FP32ImmPtr>',
        'bool': 'std::optional<BoolImmPtr>',
        'number': 'std::optional<ScalarPtr>',
        'tensor': 'std::optional<TensorPtr>',
        'str': 'std::optional<StringImmPtr>',
        'type': 'std::optional<TypePtr>',
        'tuple[int]': kValueTuplePtrOptional,
        'tuple[float]': kValueTuplePtrOptional,
        'tuple[bool]': kValueTuplePtrOptional,
    }
    if optional:
        if dtype in optional_type_convert:
            return optional_type_convert[dtype]
        raise TypeError(f"""Unsupported convert optional type {dtype} for args.""")
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")


def is_cube(class_name):
    cube_set = {'Bmm', 'Baddbmm', 'MatMul'}
    if class_name in cube_set:
        return True
    return False


def get_return_type(dtype: str):
    """
    Convert type
    """
    # add more type here
    type_convert = {
        'tuple[tensor]': 'std::vector<tensor::TensorPtr>',
        'list[tensor]': 'std::vector<tensor::TensorPtr>',
        'tensor': 'tensor::TensorPtr',
    }
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")


def get_disable_flag(yaml_def):
    """
    Get class or functional api disable generate flag.
    """
    disable_flag = False
    if yaml_def is not None:
        item = yaml_def.get("disable")
        if item is not None:
            if item is not True and item is not False:
                raise TypeError(f"The disable label for function should be True or False, but get {item}.")
            disable_flag = item
    return disable_flag


def get_op_name(operator_name, class_def):
    """
    Get op name for python class Primitive or c++ OpDef name.
    """
    class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
    if class_def is not None:
        item = class_def.get("name")
        if item is not None:
            class_name = item
    return class_name


def get_pyboost_name(operator_name):
    return 'pyboost_' + operator_name


def convert_python_func_name_to_c(func_name: str) -> str:
    return ''.join(word.capitalize() for word in func_name.split('_'))


def get_const_number_convert(arg_name, arg_type):
    cpp_type = number_input_to_cpp_type(arg_type)
    return f"auto {arg_name}_imm = GetValue<{cpp_type}>({arg_name});\n"


def get_tuple_input_convert(arg_name, arg_type):
    """
    convert tuple input.
    :param arg_name:
    :param arg_type:
    :return:
    """
    cpp_type = tuple_input_to_cpp_type(arg_type)
    return f"std::vector<{cpp_type}> {arg_name}_vector = ConvertValueTupleToVector<{cpp_type}>({arg_name});\n"


def is_pyboost_enable(operator_data):
    dispatch_key = 'dispatch'
    if dispatch_key in operator_data.keys():
        enable = operator_data[dispatch_key].get('enable')
        if enable:
            return True
    return False


class AclnnUtils:
    """
    aclnn utils
    """
    work_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../")
    aclnn_map = safe_load_yaml(os.path.join(work_path, "./mindspore/python/mindspore/ops_generate/aclnn_config.yaml"))

    @staticmethod
    def get_aclnn_interface(class_name):
        """
        get aclnn interface name.
        :param class_name:
        :return:
        """
        if class_name in AclnnUtils.aclnn_map.keys():
            return AclnnUtils.aclnn_map[class_name]
        return "aclnn" + class_name
