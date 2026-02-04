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

"""Tensor Func Proto module for defining tensor_py function prototypes and their arguments."""
import ast
import os
from collections import defaultdict

import common.gen_constants as K
from common.gen_utils import safe_load_yaml_from_dir
from resources.resource_loader import ResourceLoader
from resources.resource_list import ResourceType


class OpApiProto:
    """
    Represents a tensor function prototype with associated function name, operation prototype, and target devices.
    """

    def __init__(self,
                 func_name,
                 op_proto,
                 py_method,
                 kw_only_args,
                 varargs,
                 disable_scalar_tensor,
                 ascend,
                 gpu,
                 cpu):
        self.func_name = func_name
        self.op_proto = op_proto
        self.py_method = py_method
        self.kw_only_args = kw_only_args
        self.varargs = varargs
        self.disable_scalar_tensor = disable_scalar_tensor
        self.ascend = ascend
        self.gpu = gpu
        self.cpu = cpu


def get_tensor_method_ast_dict():
    """
        Generates a dictionary mapping function names to their Abstract Syntax Tree (AST) nodes
        for all functions defined in the 'tensor_method.py' file.
    """
    tensor_method_ast_dict = {}
    tensor_method_file = os.path.join(
        K.WORK_DIR, K.PY_MS_DIR, 'ops/tensor_method.py')
    with open(tensor_method_file, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=tensor_method_file)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            tensor_method_ast_dict[node.name] = node
    return tensor_method_ast_dict


class OpApiProtoLoader(ResourceLoader):
    """
    Loads api related proto data.
    """
    def __init__(self, op_protos, deprecated_op_protos, func_op_protos):
        self.op_api_data = safe_load_yaml_from_dir(os.path.join(K.WORK_DIR, K.MS_OP_API_YAML_PATH))
        self.op_protos = op_protos + func_op_protos
        self.deprecated_op_protos = deprecated_op_protos

    def _deal_with_varargs(self, func_data, op_name):
        """
        deal with varargs in func_data.
        """
        varargs = func_data.get('varargs', None)
        if varargs:
            varargs = [item.strip() for item in varargs.split(',')]
            check_varargs(varargs, op_name)
        return varargs

    def _deal_with_disable_scalar_tensor(self, func_data, op_name):
        """
        deal with disable_scalar_tensor in func_data.
        """
        disable_scalar_tensor = func_data.get('disable_scalar_tensor', None)
        if disable_scalar_tensor:
            disable_scalar_tensor = [item.strip() for item in disable_scalar_tensor.split(',')]
            check_disable_scalar_tensor_list(disable_scalar_tensor, op_name)
        return disable_scalar_tensor

    def _deal_with_backend(self, func_data):
        """
        deal with backend in func_data.
        """
        ascend = func_data.get('Ascend', 'aclnn')
        gpu = func_data.get('GPU', 'aclnn')
        cpu = func_data.get('CPU', 'aclnn')
        return ascend, gpu, cpu

    def _deal_with_interface(self, func_data, func_name):
        """
        deal with interface in func_data.
        """
        interface = func_data.get('interface')
        if interface is None:
            raise ValueError(
                f"For generating tensor or functional interfaces, field interface must exist. "
                f"Op name is {func_name}")

        interface = ', '.join(part.strip()
                                for part in interface.split(','))

        if interface not in {'tensor', 'function', 'tensor, function', 'function, tensor'}:
            raise ValueError(
                f"The value of field 'interface' must be one of 'tensor', 'function', "
                f"'tensor, function', or 'function, tensor'. File name is {func_name}.yaml"
            )

        return interface

    def load(self):
        """
        Loads tensor function prototypes from YAML data and returns them as a dictionary.
        """
        op_protos_dict = {}
        for op_proto in self.op_protos:
            op_protos_dict[op_proto.op_name] = op_proto
        for deprecated_op_proto in self.deprecated_op_protos:
            op_protos_dict[deprecated_op_proto.op_name] = deprecated_op_proto
        tensor_method_protos = defaultdict(list)
        mint_func_protos = defaultdict(list)
        alias_api_mapping = defaultdict(list)
        tensor_method_def_ast_dict = get_tensor_method_ast_dict()
        for func_name, tensor_func_data in self.op_api_data.items():
            func_data_list = [tensor_func_data] if isinstance(
                tensor_func_data, dict) else tensor_func_data
            for func_data in func_data_list:
                func_keys = func_data.keys()
                check_op_api_yaml_keys(func_name, set(
                    func_keys), K.TENSOR_FUNC_KEYS)
                if 'alias' in func_data:
                    alias_api_mapping[func_data['alias']].append(func_name)
                    continue
                op_name = _get_op_name_from_op_yaml(func_name, func_data)
                op_proto = op_protos_dict.get(op_name, None)
                if op_proto is None:
                    raise TypeError(
                        f"For generating tensor functions, op_proto should not be empty. Func name is {func_name}")
                py_method = func_data.get('py_method', '')
                if py_method == '':
                    raise TypeError(
                        f'For generating tensor functions, py method should not be empty. Func name is {func_name}')
                if py_method not in tensor_method_def_ast_dict:
                    raise TypeError(
                        f"{py_method} is not defined in tensor_method.py.")
                kw_only_args = func_data.get('kwonlyargs', None)
                if kw_only_args:
                    kw_only_args = [item.strip()
                                    for item in kw_only_args.split(',')]
                    check_kwonlyargs(func_data, kw_only_args, op_name,
                                     op_proto, py_method, tensor_method_def_ast_dict)
                varargs = self._deal_with_varargs(func_data, op_name)
                disable_scalar_tensor = self._deal_with_disable_scalar_tensor(func_data, op_name)
                ascend, gpu, cpu  = self._deal_with_backend(func_data)
                interface = self._deal_with_interface(func_data, func_name)

                proto = OpApiProto(func_name=func_name, op_proto=op_proto, py_method=py_method,
                                   kw_only_args=kw_only_args, varargs=varargs,
                                   disable_scalar_tensor=disable_scalar_tensor,
                                   ascend=ascend, gpu=gpu, cpu=cpu)

                if 'tensor' in interface:
                    tensor_method_protos[func_name].append(proto)
                if 'function' in interface:
                    mint_func_protos[func_name].append(proto)

        return {ResourceType.TENSOR_METHOD_PROTOS: tensor_method_protos,
                ResourceType.MINT_FUNC_PROTOS: mint_func_protos,
                ResourceType.ALIAS_API_MAPPING: alias_api_mapping}




def check_kwonlyargs(func_data, kw_only_args, op_name, op_proto, py_method, tensor_method_def_ast_dict):
    """
    Verifies that the keyword-only arguments (kwonlyargs) specified in the YAML definition
    match the order and names of the keyword-only arguments in the Python method definition.
    """
    op_args = op_proto.op_args
    kw_args_start_idx = len(op_args) - len(kw_only_args)
    node = tensor_method_def_ast_dict[py_method]
    tensor_method_kwonlyargs = [arg.arg for arg in node.args.kwonlyargs]
    for idx, kw_arg in enumerate(kw_only_args):
        kw_args_idx = kw_args_start_idx + idx
        if kw_args_idx > len(op_args) or kw_arg != op_args[kw_args_idx].arg_name:
            op_kw_args = [op_arg.arg_name for op_arg in op_args]
            op_yaml = func_data.get('op_yaml')
            raise TypeError(
                f"For generating tensor functions from {op_name}.yaml, "
                f"the order of kwonlyargs should be consistent with the definition in the {op_yaml}. "
                f"Expect kwonlyarg: {op_kw_args[kw_args_start_idx:]}, current kwonlyarg: {kw_only_args}.")
    if tensor_method_kwonlyargs != kw_only_args:
        raise TypeError(f"The order of kwonlyargs in {py_method} should be consistent with the definition. "
                        f"Expect kwonlyarg: {kw_only_args}, current kwonlyarg: {tensor_method_kwonlyargs}.")


def check_varargs(varargs, op_name):
    if len(varargs) != 1:
        raise ValueError(
            f'There must be only one variable argument. But got {len(varargs)} in {op_name}')


def check_disable_scalar_tensor_list(disable_scalar_tensor_list, op_name):
    """
    Check if the disable_scalar_tensor_list is valid.
    """
    if len(disable_scalar_tensor_list) < 1:
        raise ValueError(
            f'There must be at least one argument. But got {len(disable_scalar_tensor_list)} in {op_name}')


def _get_op_name_from_op_yaml(func_name: str, func_data: dict) -> str:
    """Extracts the operation name from the given YAML function data."""
    op_yaml = func_data.get('op_yaml', '')
    if op_yaml == '':
        raise TypeError(
            f'For generating tensor functions, op yaml should not be empty, func name is {func_name}')
    if 'deprecated' in op_yaml:
        op_name = op_yaml.replace('/', '_').replace('_method.yaml', '')
    else:
        op_name = op_yaml.replace('_op.yaml', '')
    if op_name == '':
        raise TypeError(
            f'For generating tensor functions, op name should not be empty, func name is {func_name}')
    return op_name


def check_op_api_yaml_keys(func_name: str, input_keys: set, compare_keys: set):
    diff_keys = input_keys - compare_keys
    if diff_keys:
        raise TypeError(
            f'The definition of keys in yaml has faults, func name is {func_name}, wrong keys are {diff_keys}.')


def categorize_func_data(func_protos_data):
    """
    Categorizes function prototypes into single, overloaded function prototypes.

    Args:
        func_protos_data (dict): Dictionary where keys are function API names and values are lists of
                                 function prototypes associated with each API.

    Returns:
        tuple:
            - single_op_func_data (dict): Function prototypes for operations with a single definition.
            - overload_op_func_data (dict): Function prototypes for operations with overloaded definitions.
    """
    single_op_func_data = {}
    overload_op_func_data = {}
    all_op_func_data = {}
    op_class_name_set = set()
    for func_api_name, func_protos in func_protos_data.items():
        if len(func_protos) == 1:
            func_name = func_protos[0].func_name
            if func_name not in single_op_func_data:
                single_op_func_data[func_name] = func_protos[0]
                all_op_func_data[func_name] = func_protos
        elif len(func_protos) > 1:
            overload_op_func_data[func_api_name] = func_protos
            all_op_func_data[func_api_name] = func_protos
        for func_proto in func_protos:
            op_class_name_set.add(func_proto.op_proto.op_class.name)

    return all_op_func_data, single_op_func_data, overload_op_func_data, op_class_name_set
