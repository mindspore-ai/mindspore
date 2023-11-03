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
"""Op Proto."""
from enum import Enum
from typing import List, Dict
from .pyboost_utils import convert_python_func_name_to_c


class Arg:
    def __init__(self, arg_name, arg_dtype, type_cast=[], as_init_arg=False, init=None):
        self.arg_name = arg_name
        self.arg_dtype = arg_dtype
        self.type_cast = type_cast
        self.as_init_arg = as_init_arg
        self.init = init


class OpProto:
    """
    This class defines mindspore op prototype, we parse ops.yaml to the object, to auto generate primitive
    and pyboost function.
    """

    def __init__(self):
        self.operator_name: str = ""
        self.op_name: str = ""
        self.op_args: List[Arg] = []
        self.returns: List[Arg] = []
        self.indexes: Dict[str, int] = {}
        self.pyboost_function_name: str = ""

    def update_data(
            self,
            operator_name,
            op_args,
            returns,
            class_name):
        self.operator_name = operator_name
        self.class_name = class_name
        self.op_args = op_args
        self.returns = returns
        self.indexes = {arg.arg_name: index for index, arg in enumerate(op_args)}
        self.pyboost_function_name = "Pyboost_" + self.class_name

    @staticmethod
    def load_from_yaml(op_name, yaml):
        if 'args' not in yaml.keys():
            raise TypeError("op define need key 'args'")
        args_dict = yaml.get('args')
        op_args = []
        for arg_name in args_dict.keys():
            arg_dtype = args_dict[arg_name]['dtype']
            init = None
            as_init_arg = False
            type_cast = []
            if 'init' in args_dict[arg_name]:
                init = args_dict[arg_name]['init']
                as_init_arg = True
            if 'type_cast' in args_dict[arg_name]:
                type_cast = [cast_type.strip() for cast_type in args_dict[arg_name]['type_cast'].split(',')]
            arg = Arg(arg_name, arg_dtype, type_cast, as_init_arg, init)
            op_args.append(arg)
        if 'returns' not in yaml.keys():
            raise TypeError("op define need key 'returns'")
        return_dict = yaml['returns']
        class_name = convert_python_func_name_to_c(op_name)
        if 'class' in yaml.keys() and 'name' in yaml['class'].keys():
            class_name = yaml['class']['name']
        return_args = []
        for return_name in return_dict.keys():
            dtype = return_dict[return_name]['dtype']
            arg = Arg(return_name, dtype)
            return_args.append(arg)
        op_proto = OpProto()
        op_proto.update_data(op_name, op_args, return_args, class_name)
        return op_proto
