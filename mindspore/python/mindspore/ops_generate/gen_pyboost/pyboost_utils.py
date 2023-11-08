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
        'tuple[float]': 'float',
        'tuple[bool]': 'bool',
        'tuple[tensor]': 'TensorPtr',
        'list[int]': 'int64_t',
        'list[float]': 'float',
        'list[bool]': 'bool',
        'list[tensor]': 'TensorPtr',
    }
    if dtype in types_map:
        return types_map[dtype]
    return None


def number_input_to_cpp_type(dtype: str):
    types_map = {
        'int': 'int64_t',
        'float': 'float',
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
        'float': 'FP32ImmPtr',
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
    return "auto {}_imm = GetValue<{}>({});\n".format(arg_name, number_input_to_cpp_type(arg_type), arg_name)


def get_tuple_input_convert(arg_name, arg_type):
    cpp_type = tuple_input_to_cpp_type(arg_type)
    return "std::vector<{}> {}_vector = ConvertValueTupleToVector<{}>({});\n".format(cpp_type, arg_name, cpp_type,
                                                                                     arg_name)


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
