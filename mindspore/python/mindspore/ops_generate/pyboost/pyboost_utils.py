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
import logging
from common.gen_utils import safe_load_yaml
from common.op_proto import OpProto
import common.gen_constants as K


def get_pyboost_arg_handlers_black_list():
    """
    Tensor-to-scalar conversion is already handled by ParserArgs and Converter,
    so it's not needed in PyBoost.
    :return: black list of arg handlers
    """
    black_list_for_arg_handlers = ["_normalize_int_sequence", "_scalar_tensor_to_scalar",
                                    "_scalar_tensor_to_int", "_scalar_tensor_to_float"]
    return black_list_for_arg_handlers


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


def is_op_multi_output(args):
    """
    is multi output
    :param args:
    :return: bool
    """
    if len(args) > 1:
        return True
    if len(args) == 1 and is_tensor_list(args[0]):
        return True
    return False


def get_index(index: int):
    """
    get index
    :param index:
    :return: str
    """
    return "kIndex" + str(index)


def get_convert_type_str(dtype: str, optional, use_basic_type=False):
    """
    Convert type
    """
    # add more type here
    native_type_convert = {
        'int': 'ToInt',
        'float': 'ToFloat',
        'bool': 'ToBool',
        'number': 'ToScalar',
        'tuple[int]': 'ToIntList<CPythonTuple>',
        'tuple[float]': 'ToFloatList<CPythonTuple>',
        'tuple[bool]': 'ToBoolList<CPythonTuple>',
        'tuple[tensor]': 'ToTensorList<CPythonTuple>',
        'list[int]': 'ToIntList<CPythonList>',
        'list[float]': 'ToFloatList<CPythonList>',
        'list[bool]': 'ToBoolList<CPythonList>',
        'list[tensor]': 'ToTensorList<CPythonList>',
        'tensor': 'ToTensor',
        'str': 'ToString',
        'type': 'ToDtype',
    }
    optional_type_convert = {
        'int': 'ToIntOptional',
        'float': 'ToFloatOptional',
        'number': 'ToScalarOptional',
        'tensor': 'ToTensorOptional',
        'type': 'ToDtypeOptional',
        'str': 'ToStringOptional',
        'tuple[int]': 'ToIntListOptional<CPythonTuple>',
        'tuple[float]': 'ToFloatListOptional<CPythonTuple>',
        'tuple[bool]': 'ToBoolListOptional<CPythonTuple>',
        'tuple[tensor]': 'ToTensorListOptional<CPythonTuple>',
        'list[int]': 'ToIntListOptional<CPythonList>',
        'list[float]': 'ToFloatListOptional<CPythonList>',
        'list[bool]': 'ToBoolListOptional<CPythonList>',
        'list[tensor]': 'ToTensorListOptional<CPythonList>',
    }
    basic_optional_type_convert = {
        'tuple[int]': "ToBasicIntVectorOptional<CPythonTuple>",
        'list[int]': "ToBasicIntVectorOptional<CPythonList>",
        'int': "ToBasicIntOptional",
        'type': 'ToBasicIntOptional'
    }
    basic_type_convert = {
        'tuple[int]': "ToBasicIntVector<CPythonTuple>",
        'list[int]': "ToBasicIntVector<CPythonList>",
        'int': "ToBasicInt",
        'type': 'ToBasicInt'
    }
    if optional:
        if use_basic_type and dtype in basic_optional_type_convert:
            return basic_optional_type_convert[dtype]
        if dtype in optional_type_convert:
            return optional_type_convert[dtype]
        raise TypeError(f"""Unsupported convert optional type {dtype} for args.""")
    if use_basic_type and dtype in basic_type_convert:
        return basic_type_convert[dtype]
    if dtype in native_type_convert:
        return native_type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")


def get_input_args_type_str(dtype: str, optional, use_basic_type=False):
    """
    Convert type
    """
    # add more type here
    native_type = {
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
        'tensor': 'ValuePtr',
        'str': 'StringImmPtr',
        'type': 'Int64ImmPtr',
    }
    optional_type = {
        'int': 'std::optional<Int64ImmPtr>',
        'float': 'std::optional<FP32ImmPtr>',
        'number': 'std::optional<ScalarPtr>',
        'tensor': 'std::optional<ValuePtr>',
        'type': 'std::optional<Int64ImmPtr>',
        'str': 'std::optional<StringImmPtr>',
        'tuple[int]': 'std::optional<ValueTuplePtr>',
        'tuple[float]': 'std::optional<ValueTuplePtr>',
        'tuple[bool]': 'std::optional<ValueTuplePtr>',
        'tuple[tensor]': 'std::optional<ValueTuplePtr>',
        'list[int]': 'std::optional<ValueTuplePtr>',
        'list[float]': 'std::optional<ValueTuplePtr>',
        'list[bool]': 'std::optional<ValueTuplePtr>',
        'list[tensor]': 'std::optional<ValueTuplePtr>',
    }
    basic_optional_type_convert = {
        'tuple[int]': "std::optional<std::vector<int64_t>>",
        'list[int]': "std::optional<std::vector<int64_t>>",
        'int': "std::optional<int64_t>",
        'type': "std::optional<int64_t>"
    }
    basic_type_convert = {
        'tuple[int]': "std::vector<int64_t>",
        'list[int]': "std::vector<int64_t>",
        'int': "int64_t",
        'type': "int64_t"
    }
    if optional:
        if use_basic_type and dtype in basic_optional_type_convert:
            return basic_optional_type_convert[dtype]
        if dtype in optional_type:
            return optional_type[dtype]
        raise TypeError(f"""Unknown optional type {dtype} for args.""")
    if use_basic_type and dtype in basic_type_convert:
        return basic_type_convert[dtype]
    if dtype in native_type:
        return native_type[dtype]
    raise TypeError(f"""Unknown type {dtype} for args.""")


def basic_type_convert_str(dtype: str, optional):
    """
    Convert type
    """
    optional_type = {
        'tuple[int]': "ToBasicIntVectorOptional",
        'list[int]': "ToBasicIntVectorOptional",
        'int': "ToBasicIntOptional",
        'type': "ToBasicIntOptional"
    }
    native_type = {
        'tuple[int]': "ToBasicIntVector",
        'list[int]': "ToBasicIntVector",
        'int': "ToBasicInt",
        'type': "ToBasicInt"
    }
    if optional:
        if dtype in optional_type:
            return optional_type[dtype]
    if dtype in native_type:
        return native_type[dtype]
    return ""


def get_value_convert_type_str(dtype: str, optional, use_basic_type=False):
    """
    Convert type
    """
    # add more type here
    native_type_convert = {
        'int': 'ToInt',
        'float': 'ToFloat',
        'bool': 'ToBool',
        'number': 'ToScalar',
        'tensor': 'ToTensor',
        'str': 'ToString',
        'type': 'ToDtype',
        'tuple[int]': 'ToValueTuple',
        'tuple[float]': 'ToValueTuple',
        'tuple[bool]': 'ToValueTuple',
        'tuple[tensor]': 'ToValueTuple',
        'list[int]': 'ToValueTuple',
        'list[float]': 'ToValueTuple',
        'list[bool]': 'ToValueTuple',
        'list[tensor]': 'ToValueTuple'
    }
    optional_type_convert = {
        'int': 'ToIntOptional',
        'float': 'ToFloatOptional',
        'number': 'ToScalarOptional',
        'tensor': 'ToTensorOptional',
        'type': 'ToDtypeOptional',
        'str': 'ToStringOptional',
        'tuple[int]': 'ToValueTupleOptional',
        'tuple[float]': 'ToValueTupleOptional',
        'tuple[bool]': 'ToValueTupleOptional',
        'tuple[tensor]': 'ToValueTupleOptional',
        'list[int]': 'ToValueTupleOptional',
        'list[float]': 'ToValueTupleOptional',
        'list[bool]': 'ToValueTupleOptional',
        'list[tensor]': 'ToValueTupleOptional'
    }
    basic_optional_type_convert = {
        'tuple[int]': "ToBasicIntVectorOptional",
        'list[int]': "ToBasicIntVectorOptional",
        'int': "ToBasicIntOptional",
        'type': "ToBasicIntOptional"
    }
    basic_type_convert = {
        'tuple[int]': "ToBasicIntVector",
        'list[int]': "ToBasicIntVector",
        'int': "ToBasicInt",
        'type': "ToBasicInt"
    }
    if optional:
        if use_basic_type and dtype in basic_optional_type_convert:
            return basic_optional_type_convert[dtype]
        if dtype in optional_type_convert:
            return optional_type_convert[dtype]
        raise TypeError(f"""Unsupported convert optional type {dtype} for args.""")
    if use_basic_type and dtype in basic_type_convert:
        return basic_type_convert[dtype]
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
        'tuple[str]': 'std::string',
        'tuple[tensor]': 'mindspore::tensor::TensorPtr',
        'list[int]': 'int64_t',
        'list[float]': 'float',
        'list[bool]': 'bool',
        'list[tensor]': 'mindspore::tensor::TensorPtr',
    }
    return types_map.get(dtype)


def number_input_to_cpp_type(dtype: str):
    types_map = {
        'int': 'int64_t',
        'float': 'float',
        'bool': 'bool',
        'str': 'std::string'
    }
    return types_map.get(dtype)


def input_dtype_to_cpp_type(dtype: str, optional):
    """
    Map input dtype to cpp dtype
    """
    type_convert = {
        'int': 'int64_t',
        'float': 'float',
        'bool': 'bool',
        'number': 'mindspore::ScalarPtr',
        'str': 'std::string',
        'tensor': 'mindspore::tensor::TensorPtr',
        'tuple[int]': 'std::vector<int64_t>',
        'tuple[float]': 'std::vector<float>',
        'tuple[bool]': 'std::vector<bool>',
        'tuple[tensor]': 'std::vector<mindspore::tensor::TensorPtr>',
        'list[int]': 'std::vector<int64_t>',
        'list[float]': 'std::vector<float>',
        'list[bool]': 'std::vector<bool>',
        'list[tensor]': 'std::vector<mindspore::tensor::TensorPtr>',
    }
    optional_tensor_type_convert = {
        'tensor': 'std::optional<mindspore::tensor::TensorPtr>',
        'tuple[tensor]': 'std::vector<mindspore::tensor::TensorPtr>',
        'list[tensor]': 'std::vector<mindspore::tensor::TensorPtr>'
    }
    if optional and dtype in optional_tensor_type_convert:
        return optional_tensor_type_convert[dtype]
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported dtype {dtype} for args.""")


def get_input_dtype(dtype: str, optional, use_basic_type=False):
    """
    Convert type
    """
    # add more type here
    value_tuple = 'mindspore::ValueTuplePtr'
    type_convert = {
        'int': 'mindspore::Int64ImmPtr',
        'float': 'mindspore::FP32ImmPtr',
        'bool': 'mindspore::BoolImmPtr',
        'number': 'mindspore::ScalarPtr',
        'str': 'mindspore::StringImmPtr',
        'tensor': 'mindspore::tensor::TensorPtr',
        'tuple[int]': value_tuple,
        'tuple[float]': value_tuple,
        'tuple[bool]': value_tuple,
        'tuple[tensor]': value_tuple,
        'list[int]': value_tuple,
        'list[float]': value_tuple,
        'list[bool]': value_tuple,
        'list[tensor]': value_tuple,
    }
    value_tuple_optional = 'std::optional<mindspore::ValueTuplePtr>'
    optional_type_convert = {
        'int': 'std::optional<mindspore::Int64ImmPtr>',
        'float': 'std::optional<mindspore::FP32ImmPtr>',
        'bool': 'std::optional<mindspore::BoolImmPtr>',
        'number': 'std::optional<mindspore::ScalarPtr>',
        'str': 'std::optional<mindspore::StringImmPtr>',
        'tensor': 'std::optional<mindspore::tensor::TensorPtr>',
        'tuple[int]': value_tuple_optional,
        'tuple[float]': value_tuple_optional,
        'tuple[bool]': value_tuple_optional,
        'tuple[tensor]': value_tuple_optional,
    }
    basic_optional_type_convert = {
        'tuple[int]': "std::optional<std::vector<int64_t>>",
        'list[int]': "std::optional<std::vector<int64_t>>",
        'int': "std::optional<int64_t>",
    }
    basic_type_convert = {
        'tuple[int]': "std::vector<int64_t>",
        'list[int]': "std::vector<int64_t>",
        'int': "int64_t",
    }
    if optional:
        if use_basic_type and dtype in basic_optional_type_convert:
            return basic_optional_type_convert[dtype]
        if dtype in optional_type_convert:
            return optional_type_convert[dtype]
        raise TypeError(f"""Unsupported convert optional type {dtype} for args.""")
    if use_basic_type and dtype in basic_type_convert:
        return basic_type_convert[dtype]
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")


def get_output_dtype(dtype: str):
    type_convert = {
        'tensor': "mindspore::tensor::TensorPtr",
        'tuple[tensor]': "std::vector<mindspore::tensor::TensorPtr>",
        'list[tensor]': "std::vector<mindspore::tensor::TensorPtr>",
    }
    if dtype in type_convert:
        return type_convert[dtype]
    raise TypeError(f"""Unsupported convert type {dtype} for args.""")


def is_cube(class_name):
    cube_set = {'Bmm', 'Baddbmm', 'MatMulExt', 'Mv'}
    if class_name in cube_set:
        return True
    return False


def get_return_type(dtype: str):
    """
    Convert type
    """
    # add more type here
    type_convert = {
        'tuple[tensor]': 'std::vector<mindspore::tensor::TensorPtr>',
        'list[tensor]': 'std::vector<mindspore::tensor::TensorPtr>',
        'tensor': 'mindspore::tensor::TensorPtr',
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
    if class_def:
        return class_def

    class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
    return class_name


def get_pyboost_name(operator_name):
    return 'pyboost_' + operator_name


def get_const_number_convert(arg_name, op_arg):
    cpp_type = number_input_to_cpp_type(op_arg.arg_dtype)
    if op_arg.is_type_id:
        return f"TypeId {arg_name}_imm = static_cast<TypeId>(GetValue<{cpp_type}>({arg_name}));\n"
    return f"auto {arg_name}_imm = GetValue<{cpp_type}>({arg_name});\n"


def get_tuple_input_convert(arg_name, arg_type):
    """
    convert tuple input.
    :param arg_name:
    :param arg_type:
    :return:
    """
    cpp_type = tuple_input_to_cpp_type(arg_type)
    if cpp_type == "mindspore::tensor::TensorPtr":
        cpp_type = "mindspore::tensor::TensorPtr"
    return f"std::vector<{cpp_type}> {arg_name}_vector = ConvertValueTupleToVector<{cpp_type}>({arg_name});\n"


def is_pyboost_enable(operator_data):
    dispatch_key = 'dispatch'
    if dispatch_key in operator_data.keys():
        enable = operator_data[dispatch_key].get('enable')
        if enable:
            return True
    return False


def format_func_api_name(func_api_name):
    """
    Converts a snake_case string to PascalCase format with the first letter capitalized.
    Additionally, it preserves the trailing underscore. In special cases, such as double
    underscore names (e.g., __add__), it converts them into PascalCase.

    Args:
        func_api_name (str): The input snake_case string.

    Returns:
        str: The converted PascalCase string.
    """
    # Check if the string ends with '_'
    is_one_underscore = func_api_name.endswith('_')

    # Check if it is a double-underscore name (special method names)
    is_double_underscore = func_api_name.startswith('__') and func_api_name.endswith('__')

    # If it is a double-underscore name, remove the leading and trailing underscores
    if is_double_underscore:
        func_api_name = func_api_name[2:-2]

    # If the original name ends with '_' but is not a double-underscore name, remove the trailing '_'
    if is_one_underscore and not is_double_underscore:
        func_api_name = func_api_name[:-1]

    # Convert snake_case to PascalCase
    formatted_func_api_name = ''.join(x.capitalize() for x in func_api_name.split('_'))

    # If the original name ends with '_' but is not a double-underscore name, append the trailing underscore
    if is_one_underscore and not is_double_underscore:
        formatted_func_api_name += '_'

    # If the original name is a double-underscore name, add a 'Magic' suffix.
    if is_double_underscore:
        formatted_func_api_name += 'Magic'

    return formatted_func_api_name


def convert_types(inputs):
    '''convert type to acl type'''
    inputs_dtypes = {}
    flag = False
    for i in inputs:
        inputs_dtypes[i] = i.arg_dtype
        if inputs_dtypes[i] != 'tensor':
            flag = True
        if 'tuple' in inputs_dtypes[i]:
            data_type = inputs_dtypes[i].split('[')[1].strip(']')
            if data_type == 'tensor':
                logging.info("Not support tuple[tensor] input.")
            elif data_type == 'int':
                inputs_dtypes[i] = 'std::vector<int64_t>'
            elif data_type == 'float':
                inputs_dtypes[i] = 'std::vector<float>'
            elif data_type == 'bool':
                inputs_dtypes[i] = 'std::vector<uint8_t>'
            else:
                logging.warning("Not support tuple[%s]] input.", data_type)
        if inputs_dtypes[i] == 'number':
            inputs_dtypes[i] = 'ScalarPtr'
        if inputs_dtypes[i] == 'int':
            inputs_dtypes[i] = 'int64_t'
    return inputs_dtypes, flag


def get_dtypes(op_proto: OpProto):
    """get op inputs and outputs dtypes"""
    inputs = op_proto.op_args
    outputs = op_proto.op_returns
    inputs_dtypes, flag_in = convert_types(inputs)
    outputs_dtypes, flag_out = convert_types(outputs)
    none_tensor_exist = (flag_in or flag_out)
    return inputs_dtypes, outputs_dtypes, none_tensor_exist


def merge_strings_by_chunk_size(string_list, chunk_size=50):
    """
    Merges a list of strings into smaller chunks, with each chunk having a specified maximum size.

    Args:
        string_list (list of str): A list of strings to be merged.
        chunk_size (int, optional): The maximum size of each merged chunk. Defaults to 50.

    Returns:
        list of str: A list of merged strings, where each string contains up to `chunk_size` characters.

    Example:
        >>> strings = ["Hello", "world", "this", "is", "a", "test"]
        >>> merge_strings_by_chunk_size(strings, chunk_size=2)
        ['Helloworld', 'thisis', 'atest']
    """
    merged_strings = [
        "".join(string_list[i:i + chunk_size])  # Merge the current grouped string
        for i in range(0, len(string_list), chunk_size)
    ]
    return merged_strings


def chunk_list(lst, n):
    """
    Divide a list into sublists of length 'n'.

    This function takes a list `lst` and an integer `n`, and returns a new list
    where each element is a sublist of `lst` containing up to `n` elements.
    If the length of `lst` is not a multiple of `n`, the last sublist will contain
    fewer than `n` elements.

    Args:
        lst (list): The original list to be divided.
        n (int): The number of elements per sublist.

    Returns:
        list: A list of sublists, where each sublist has up to `n` elements.

    Raises:
        ValueError: If `n` is less than or equal to 0.

    Example:
        >>> my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> chunked_list = chunk_list(my_list, 3)
        >>> print(chunked_list)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    Note:
        This function assumes that `n` is a positive integer. If `n` is not a
        positive integer, a ValueError is raised.
    """
    if n <= 0:
        raise ValueError("The chunk size 'n' must be a positive integer.")

    return [lst[i:i + n] for i in range(0, len(lst), n)]


class AclnnUtils:
    """
    aclnn utils
    """
    aclnn_map = safe_load_yaml(os.path.join(
        K.WORK_DIR, K.PY_OPS_GEN_PATH, "pyboost/aclnn_config.yaml"))

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
