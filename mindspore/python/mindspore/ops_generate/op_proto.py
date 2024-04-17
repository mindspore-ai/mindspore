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
from pyboost_utils import convert_python_func_name_to_c


class Arg:
    def __init__(self, arg_name, arg_dtype, type_cast, is_type_id=False, as_init_arg=False, default=-1, inplace=''):
        self.arg_name = arg_name
        self.arg_dtype = arg_dtype
        self.type_cast = type_cast
        self.is_type_id = is_type_id
        self.as_init_arg = as_init_arg
        self.default = default
        self.inplace = inplace


class OpProto:
    """
    This class defines mindspore op prototype, we parse ops.yaml to the object, to auto generate primitive
    and pyboost function.
    """

    def __init__(self,
                 operator_name,
                 op_args,
                 returns,
                 class_name,
                 is_pyboost,
                 is_view,
                 cpu,
                 gpu,
                 ascend,
                 prim_init,
                 is_dispatch):
        self.operator_name = operator_name
        self.class_name = class_name
        self.op_args = op_args
        self.returns = returns
        self.indexes = {arg.arg_name: index for index, arg in enumerate(op_args)}
        self.pyboost_function_name = "Pyboost_" + self.class_name
        self.is_pyboost = is_pyboost
        self.is_view = is_view
        self.cpu = cpu
        self.gpu = gpu
        self.ascend = ascend
        self.prim_init = prim_init
        self.is_dispatch = is_dispatch

    @staticmethod
    def get_device_special_name(dispatch, gpu, cpu, ascend):
        if 'GPU' in dispatch.keys():
            gpu = dispatch['GPU']
        if 'CPU' in dispatch.keys():
            cpu = dispatch['CPU']
        if 'Ascend' in dispatch.keys():
            ascend = dispatch['Ascend']
        return gpu, cpu, ascend

    @staticmethod
    def load_from_yaml(op_name, yaml):
        """
        load from yaml
        :param op_name:
        :param yaml:
        :return:
        """
        if 'args' not in yaml.keys():
            raise TypeError("op define need key 'args'")
        args_dict = yaml.get('args')
        op_args = []
        default_str = 'default'
        is_type_id = False
        prim_init = False
        for arg_name in args_dict.keys():
            arg_dtype = args_dict[arg_name]['dtype']
            if arg_dtype == 'TypeId':
                arg_dtype = 'int'
            default = None
            as_init_arg = False
            is_type_id = False
            type_cast = []
            if default_str in args_dict[arg_name]:
                default = args_dict[arg_name][default_str]
                as_init_arg = True
            if 'prim_init' in args_dict[arg_name]:
                prim_init = args_dict[arg_name]['prim_init']
            if 'type_cast' in args_dict[arg_name]:
                type_cast = [cast_type.strip() for cast_type in args_dict[arg_name]['type_cast'].split(',')]
            arg_handler_key = 'arg_handler'
            if arg_handler_key in args_dict[arg_name] and args_dict[arg_name][arg_handler_key] == 'dtype_to_type_id':
                is_type_id = True
            arg = Arg(arg_name, arg_dtype, type_cast, is_type_id, as_init_arg, default)
            op_args.append(arg)
        if 'returns' not in yaml.keys():
            raise TypeError("op define need key 'returns'")

        is_pyboost = False
        is_dispatch = False
        gpu = default_str
        cpu = default_str
        ascend = default_str
        dispatch_key = 'dispatch'
        if dispatch_key in yaml.keys():
            is_dispatch = True
            is_pyboost = yaml[dispatch_key].get('enable')
            gpu, cpu, ascend = OpProto.get_device_special_name(yaml[dispatch_key], gpu, cpu, ascend)
        return_dict = yaml['returns']
        class_name = convert_python_func_name_to_c(op_name)
        class_key = 'class'
        if class_key in yaml.keys() and 'name' in yaml[class_key].keys():
            class_name = yaml[class_key]['name']
        return_args = []
        for return_name in return_dict.keys():
            inplace = ''
            if 'inplace' in return_dict[return_name]:
                inplace = return_dict[return_name]['inplace']
            dtype = return_dict[return_name]['dtype']
            arg = Arg(return_name, dtype, type_cast=[], inplace=inplace)
            return_args.append(arg)
        is_view = False
        if 'view' in yaml.keys():
            is_view = True
        op_proto = OpProto(op_name, op_args, return_args, class_name,
                           is_pyboost, is_view, cpu, gpu, ascend, prim_init, is_dispatch)
        return op_proto
