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

"""Op Proto module for defining operator prototypes and their arguments."""

import os
from typing import Dict, List

from resources.resource_loader import ResourceLoader
from resources.resource_list import ResourceType

from . import gen_constants as K
from .gen_utils import safe_load_yaml_from_dir


class OpArg:
    """
    Represents an argument of an operator.

    Attributes:
        arg_name (str): The name of the argument.
        arg_dtype (str): The data type of the argument.
        type_cast (set): A set of type casts applicable to the argument.
        is_type_id (bool): Indicates if the argument is a type identifier.
        as_init_arg (bool): Indicates if the argument is an initialization argument.
        default: The default value of the argument.
        inplace (str): The name of the inplace tensor if applicable.
        is_prim_init (bool): Indicates if the argument is a primitive initialization argument.
        arg_handler (str): A handler for the argument, if applicable.
    """

    def __init__(self, arg_name, arg_dtype, type_cast, is_type_id=False, as_init_arg=False, default=-1, inplace='',
                 is_prim_init=False, arg_handler=''):
        self.arg_name = arg_name
        self.arg_dtype = arg_dtype
        self.type_cast = type_cast
        self.is_type_id = is_type_id
        self.as_init_arg = as_init_arg
        self.default = default
        self.inplace = inplace
        self.is_prim_init = is_prim_init
        self.arg_handler = arg_handler


class OpArgsSignature:
    """
    Represents the signature of operator arguments.

    Attributes:
        rw_write (list): Arguments that are written to.
        rw_read (list): Arguments that are read from.
        rw_ref (list): Arguments that are passed by reference.
        dtype_group (list): Grouping of data types for the arguments.
    """

    def __init__(self, rw_write=None, rw_read=None, rw_ref=None, dtype_group=None):
        self.rw_write = rw_write
        self.rw_read = rw_read
        self.rw_ref = rw_ref
        self.dtype_group = dtype_group


class OpFunction:
    """
    Represents the function associated with an operator.

    Attributes:
        disable (bool): Indicates if the function is disabled.
        name (str): The name of the function.
    """

    def __init__(self, disable=False, name=''):
        self.disable = disable
        self.name = name


class OpClass:
    """
    Represents a class associated with an operator.

    Attributes:
        disable (bool): Indicates if the class is disabled.
        name (str): The name of the class.
    """

    def __init__(self, disable=False, name=''):
        self.disable = disable
        self.name = name


class OpDispatch:
    """
    Represents the dispatch information for an operator.

    Attributes:
        enable (bool): Indicates if the dispatch is enabled.
        is_comm_op (bool): Indicates if the dispatch is communication operator or not.
        ascend (str): The dispatch type for the Ascend device.
        cpu (str): The dispatch type for the CPU.
        gpu (str): The dispatch type for the GPU.
    """

    def __init__(self, enable=False, is_comm_op=False, ascend='default',
                 internal_op_ascend='None', cpu='default', gpu='default'):
        self.enable = enable
        self.is_comm_op = is_comm_op
        self.ascend = ascend
        self.internal_op_ascend = internal_op_ascend
        self.cpu = cpu
        self.gpu = gpu


class OpProto:
    """
    Defines a prototype for an operator in MindSpore.

    This class is used to parse the operator definition from a YAML file and to generate
    the necessary primitive and PyBoost functions.

    Attributes:
        op_name (str): The name of the operator.
        op_args (list): A list of arguments for the operator.
        op_function (OpFunction): The function associated with the operator.
        op_class (OpClass): The class associated with the operator.
        op_dispatch (OpDispatch): The dispatch information for the operator.
        op_args_signature (OpArgsSignature): The signature of the operator's arguments.
        op_returns (list): A list of return values for the operator.
        op_view (bool): Indicates if the operator is a view operator.
    """

    def __init__(self,
                 op_name: str,
                 op_args: List[OpArg],
                 op_function: OpFunction,
                 op_class: OpClass,
                 op_dispatch: OpDispatch,
                 op_args_signature: OpArgsSignature,
                 op_returns: List[OpArg],
                 op_view: bool = False,
                 op_graph_view=False,
                 op_inplace=False,
                 op_labels=None,
                 op_deprecated=None,
                 bprop_expander=True,
                 non_differentiable=False,
                 composite=False):
        self.op_name = op_name
        self.op_args = op_args
        self.op_function = op_function
        self.op_class = op_class
        self.op_dispatch = op_dispatch
        self.op_args_signature = op_args_signature
        self.op_returns = op_returns
        self.op_view = op_view
        self.op_graph_view = op_graph_view
        self.op_inplace = op_inplace
        self.op_labels = op_labels
        self.op_deprecated = op_deprecated
        self.bprop_expander = bprop_expander
        self.non_differentiable = non_differentiable
        self.composite = composite

    @staticmethod
    def load_from_yaml(op_name, op_data):
        """
        Loads an operator prototype from YAML data.

        Args:
            op_name (str): The name of the operation.
            op_data (dict): A dictionary containing the operation data.

        Returns:
            OpProto: An instance of OpProto representing the operator.
        """
        # check op keys
        check_validation(op_name, op_data)
        # get op args
        op_args = get_op_args(op_name, op_data)
        # get op return args
        op_returns = get_op_returns(op_name, op_data)
        # get op args signature
        op_args_signature = get_op_args_signature(op_name, op_data)
        # get op class
        op_class = get_op_class(op_name, op_data)
        # get op function
        op_function = get_op_function(op_name, op_data)
        # get op dispatch
        op_dispatch = get_op_dispatch(op_name, op_data)
        # get op view
        op_view = op_data.get('view', False)
        if not isinstance(op_view, bool):
            raise TypeError(
                f'The view value should be bool, but get {type(op_view)}, op name is {op_name}.')
        # get op graph view
        op_graph_view = op_data.get('graph_view', False)
        if not isinstance(op_graph_view, bool):
            raise TypeError(
                f'The graph view value should be bool, but get {type(op_graph_view)}, op name is {op_name}.')
        op_inplace = is_inplace_op(op_returns)
        # get op labels
        op_labels = op_data.get('labels', None)
        # get op deprecated
        op_deprecated = op_data.get('deprecated', None)
        bprop_expander = op_data.get('bprop_expander', True)
        composite = op_data.get('composite', False)
        non_differentiable = op_data.get('non-differentiable', False)
        op_proto = OpProto(op_name=op_name, op_args=op_args, op_returns=op_returns, op_function=op_function,
                           op_class=op_class, op_dispatch=op_dispatch, op_args_signature=op_args_signature,
                           op_view=op_view, op_graph_view=op_graph_view, op_inplace=op_inplace, op_labels=op_labels,
                           op_deprecated=op_deprecated, bprop_expander=bprop_expander,
                           non_differentiable=non_differentiable, composite=composite)
        return op_proto


class OpProtoLoader(ResourceLoader):
    """
    OpProtoLoader is a class for loading operator prototypes from YAML data.
    """

    def __init__(self):
        ops_yaml_path = os.path.join(K.WORK_DIR, K.MS_OP_DEF_YAML_PATH)
        infer_ops_yaml_path = os.path.join(ops_yaml_path, 'infer')
        comm_ops_yaml_path = os.path.join(ops_yaml_path, 'communication')
        self.yaml_paths = [ops_yaml_path, infer_ops_yaml_path, comm_ops_yaml_path]
        self.type = ResourceType.OP_PROTO
        self.is_deprecated = False
        self.func_op = False

    def load(self) -> Dict[ResourceType, object]:
        """
        Load OpProto.

        Returns:
            Dict[ResourceType, object]: The resource type and the OpProto.
        """
        yaml_dict = {}
        for yaml_path in self.yaml_paths:
            yaml_dict.update(safe_load_yaml_from_dir(yaml_path))
        op_protos = []
        for op_name, op_data in yaml_dict.items():
            op_proto = OpProto.load_from_yaml(op_name, op_data)
            if self.is_deprecated:
                op_proto.op_name = 'deprecated_' + op_name
            op_proto.func_op = self.func_op
            op_protos.append(op_proto)
        return {self.type: op_protos}


class DeprecatedOpProtoLoader(OpProtoLoader):
    """
    DeprecatedOpProtoLoader is a class for loading deprecated operator prototypes from YAML data.
    """

    def __init__(self):
        super().__init__()
        self.yaml_paths = [os.path.join(K.WORK_DIR, K.MS_OP_DEPRECATED_DEF_YAML_PATH)]
        self.type = ResourceType.DEPRECATED_OP_PROTO
        self.is_deprecated = True
        self.func_op = True


class FuncOpProtoLoader(OpProtoLoader):
    """
    FuncOpProtoLoader is a class for loading func_op operator prototypes from YAML data.
    """

    def __init__(self):
        super().__init__()
        self.yaml_paths = [os.path.join(K.WORK_DIR, K.MS_OP_DEF_FUNC_OP_YAML_PATH)]
        self.type = ResourceType.FUNC_OP_PROTO
        self.is_deprecated = False
        self.func_op = True


class CustomOpProtoLoader(OpProtoLoader):
    """
    CustomOpProtoLoader is a class for loading custom_op operator prototypes from YAML data.
    """

    def __init__(self, yaml_dir_path):
        super().__init__()
        self.yaml_paths = [yaml_dir_path]
        self.type = ResourceType.OP_PROTO
        self.is_deprecated = False
        self.func_op = False


def get_op_args_signature(op_name, op_data):
    """
    Retrieves the argument signature from the operation data.

    Args:
        op_data (dict): A dictionary containing the operation data.

    Returns:
        OpArgsSignature: An instance of OpArgsSignature containing the argument signature.
    """
    op_args_signature = op_data.get('args_signature', None)
    if op_args_signature is not None:
        args_signature_keys = op_args_signature.keys()
        check_op_yaml_keys(op_name, set(args_signature_keys), K.ARG_SIGNATURE_KEYS)
        rw_write = op_args_signature.get('rw_write', None)
        rw_read = op_args_signature.get('rw_read', None)
        rw_ref = op_args_signature.get('rw_ref', None)
        dtype_group = op_args_signature.get('dtype_group', None)
        return OpArgsSignature(rw_write, rw_read, rw_ref, dtype_group)
    return None


def check_validation(op_name: str, op_data: dict):
    """
    Validates the operator data to ensure it contains necessary keys.

    Args:
        op_data (dict): The operator data to validate.

    Raises:
        TypeError: If the required keys 'args' or 'returns' are missing.
    """
    # check keys
    check_op_yaml_keys(op_name, set(op_data.keys()), K.OP_KEYS)

    # Those keys must in yaml
    if 'args' not in op_data.keys():
        raise TypeError(f"Op define miss key 'args', op name is {op_name}")
    if 'returns' not in op_data.keys():
        raise TypeError(f"Op define miss key 'returns', op name is {op_name}")


def get_op_args(op_name: str, op_data: dict) -> List[OpArg]:
    """
    Retrieves the arguments for the operator from the operation data.

    Args:
        op_data (dict): A dictionary containing the operation data.

    Returns:
        list: A list of OpArg instances representing the arguments of the operator.
    """
    args_dict = op_data.get('args')
    op_args = []
    for arg_name in args_dict.keys():
        arg_keys = args_dict[arg_name].keys()
        check_op_yaml_keys(op_name, set(arg_keys), K.ARG_KEYS)
        arg_dtype = args_dict[arg_name]['dtype']
        if arg_dtype == 'TypeId':
            arg_dtype = 'int'
        default = None
        as_init_arg = False
        is_type_id = False
        prim_init = False
        type_cast = set()
        if 'default' in args_dict[arg_name]:
            default = args_dict[arg_name]['default']
            as_init_arg = True
        # 当op_args任意一个参数有prim_init，该op就要在pyboost_inner_prim.py生成
        if 'prim_init' in args_dict[arg_name] and args_dict[arg_name]['prim_init'] is True:
            prim_init = True
        if 'type_cast' in args_dict[arg_name]:
            type_cast = set(cast_type.strip() for cast_type in args_dict[arg_name]['type_cast'].split(','))
        arg_handler_key = 'arg_handler'
        arg_handler = args_dict[arg_name].get(arg_handler_key, '')
        if arg_handler_key in args_dict[arg_name] and args_dict[arg_name][arg_handler_key] == 'dtype_to_type_id':
            is_type_id = True

        disable_tensor_to_scalar = args_dict[arg_name].get('disable_tensor_to_scalar', False)
        # add default support of tensor to scalar
        if arg_dtype in ("tuple[int]", "list[int]") and not arg_handler:
            arg_handler = '_normalize_int_sequence'
        elif 'tensor' not in type_cast and not arg_handler and not disable_tensor_to_scalar:
            # when type_cast: tensor is specified, single-element tensor cast to scalar is supported
            # by default, only 0-dim tensor is supported to cast to scalar
            if arg_dtype == 'int':
                arg_handler = '_scalar_tensor_to_int'
            elif arg_dtype == 'float':
                arg_handler = '_scalar_tensor_to_float'
            elif arg_dtype == 'number':
                arg_handler = '_scalar_tensor_to_scalar'

        op_arg = OpArg(arg_name, arg_dtype, type_cast, is_type_id, as_init_arg, default,
                       is_prim_init=prim_init, arg_handler=arg_handler)
        op_args.append(op_arg)
    return op_args


def get_op_returns(op_name, op_data):
    """
    Retrieves the return values for the operator from the operation data.

    Args:
        op_data (dict): A dictionary containing the operation data.

    Returns:
        list: A list of OpArg instances representing the return values of the operator.
    """
    op_return_args = []
    return_dict = op_data['returns']
    for return_name in return_dict.keys():
        return_keys = return_dict[return_name].keys()
        check_op_yaml_keys(op_name, set(return_keys), K.RETURN_KEYS)
        inplace = ''
        if 'inplace' in return_dict[return_name]:
            inplace = return_dict[return_name]['inplace']
        if 'dtype' not in return_dict[return_name]:
            raise TypeError("op return args need key 'dtype'")
        dtype = return_dict[return_name]['dtype']
        arg = OpArg(return_name, dtype, type_cast=set(), inplace=inplace)
        op_return_args.append(arg)
    return op_return_args


def get_op_dispatch(op_name, op_data):
    """
    Retrieves the dispatch information for the operator from the operation data.

    Args:
        op_data (dict): A dictionary containing the operation data.

    Returns:
        OpDispatch: An instance of OpDispatch containing the dispatch information.
    """
    op_dispatch = op_data.get('dispatch', {})
    dispatch_keys = op_dispatch.keys()
    check_op_yaml_keys(op_name, set(dispatch_keys), K.DISPATCH_KEYS)
    if not op_dispatch:
        return None
    enable = op_dispatch.get('enable', False)
    if not isinstance(enable, bool):
        raise TypeError(
            f'The dispatch enable value should be bool, but get {type(enable)}, op name is {op_name}.')
    is_comm_op = op_dispatch.get('is_comm_op', False)
    ascend = op_dispatch.get('Ascend', 'default')
    internal_op_ascend = op_dispatch.get('InternalOpAscend', 'None')
    cpu = op_dispatch.get('CPU', 'default')
    gpu = op_dispatch.get('GPU', 'default')
    return OpDispatch(enable, is_comm_op, ascend, internal_op_ascend, cpu, gpu)


def get_op_class(op_name, op_data) -> OpClass:
    """
    Retrieves the class information for the operator from the operation data.

    Args:
        op_name (str): The name of the operation.
        op_data (dict): A dictionary containing the operation data.

    Returns:
        OpClass: An instance of OpClass containing the class information for the operator.
    """
    op_class = op_data.get('class', {})
    class_keys = op_class.keys()
    check_op_yaml_keys(op_name, set(class_keys), K.CLASS_KEYS)
    is_disable = op_class.get('disable', False)
    if not isinstance(is_disable, bool):
        raise TypeError(
            f'The class disable value should be bool, but get {type(is_disable)}, op name is {op_name}.')
    class_name = op_class.get('name', convert_python_func_name_to_c(op_name))
    return OpClass(disable=is_disable, name=class_name)


def get_op_function(op_name, op_data) -> OpFunction:
    """
    Retrieves the function information for the operator from the operation data.

    Args:
        op_name (str): default operation function name.
        op_data (dict): A dictionary containing the operation data.

    Returns:
        OpFunction: An instance of OpFunction containing the function information for the operator.
    """
    op_function = op_data.get('function', {})
    function_keys = op_function.keys()
    check_op_yaml_keys(op_name, set(function_keys), K.FUNCTION_KEYS)
    is_disable = op_function.get('disable', False)
    if not isinstance(is_disable, bool):
        raise TypeError(
            f'The function disable value should be bool, but get {type(is_disable)}, op name is {op_name}.')
    function_name = op_function.get('name', op_name)
    return OpFunction(disable=is_disable, name=function_name)


def convert_python_func_name_to_c(func_name: str) -> str:
    return ''.join(word.capitalize() for word in func_name.split('_'))


def check_op_yaml_keys(op_name: str, input_keys: set, compare_keys: set):
    diff_keys = input_keys - compare_keys
    if diff_keys:
        raise TypeError(
            f'The definition of keys in yaml has faults, op name is {op_name}, wrong keys are {diff_keys}.')


def is_inplace_op(args):
    """
    is inplace op
    :param args:
    :return: bool
    """
    for arg in args:
        if arg.inplace:
            return True
    return False
