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
"""
Module for generating C++ operator definition files.
"""

import os
import math

import common.gen_constants as K
import common.gen_utils as gen_utils

# refactored
from common.op_proto import OpProto
import common.template_utils as template

from common.base_generator import BaseGenerator

CC_OPS_DEF = """

#include "primitive/auto_generate/gen_ops_def.h"
#include "ir/signature.h"
$gen_include

namespace mindspore::ops {$gen_cc_code
}  // namespace mindspore::ops
"""


class OpsDefCcGenerator(BaseGenerator):
    """
    Generates C++ definition files for operators.
    """

    def __init__(self):
        """
        Initializes templates for generating C++ operator definitions.
        """
        self.include_template = template.Template("""#include "${path}/${operator_name}.h\"\n""")
        self.func_impl_declaration_template = template.Template("${class_name}FuncImpl g${class_name}FuncImpl;")
        self.empty_func_impl_declaration_template = template.Template("static OpFuncImpl g${class_name}FuncImpl;")
        self.func_impl_define_template = template.Template("g${class_name}FuncImpl")
        self.OP_PROTO_TEMPLATE = template.OP_PROTO_TEMPLATE
        self.CC_OPS_DEF_TEMPLATE = template.Template(CC_OPS_DEF)

    def generate(self, work_path, op_protos):
        """
        Generates C++ code for operator definitions and saves it to a file.

        Args:
            work_path (str): The directory to save the generated files.
            op_protos (list): A list of operator prototypes.
        """
        gen_cc_list = list()
        gen_include_list = list()
        gen_deprecated_cc_list = list()

        for op_proto in op_protos:
            operator_name = op_proto.op_name
            class_name = op_proto.op_class.name
            if not op_proto.func_op:
                if op_proto.op_dispatch and op_proto.op_dispatch.is_comm_op:
                    gen_include_list.append(self.include_template.replace(path=K.MS_OPS_COMM_FUNC_IMPL_PATH,
                                                                          operator_name=operator_name))
                else:
                    gen_include_list.append(self.include_template.replace(path=K.MS_OPS_FUNC_IMPL_PATH,
                                                                          operator_name=operator_name))
                func_impl_declaration_str = self.func_impl_declaration_template.replace(class_name=class_name)
            else:
                func_impl_declaration_str = self.empty_func_impl_declaration_template.replace(class_name=class_name)
            func_impl_define = self.func_impl_define_template.replace(class_name=class_name)

            # process input
            args_dict, cc_index_str, input_args_str = process_input_args(op_proto)

            # Process outputs.
            return_args_str = get_cc_op_def_return(args_dict, op_proto)

            inputs_args = self.process_args(op_proto.op_args)
            signature_code = generate_cc_op_signature(op_proto.op_args_signature, inputs_args)
            enable_dispatch = "true" if op_proto.op_dispatch and op_proto.op_dispatch.enable else "false"
            is_view = "true" if op_proto.op_view else "false"
            is_graph_view = "true" if op_proto.op_graph_view else "false"
            op_def_cc = self.OP_PROTO_TEMPLATE.replace(class_name=class_name,
                                                       input_args=input_args_str,
                                                       return_args=return_args_str,
                                                       signatures=signature_code,
                                                       indexes=cc_index_str,
                                                       enable_dispatch=enable_dispatch,
                                                       is_view=is_view,
                                                       is_graph_view=is_graph_view,
                                                       func_impl_declaration=func_impl_declaration_str,
                                                       func_impl_define=func_impl_define)

            if "deprecated" not in operator_name:
                gen_cc_list.append(op_def_cc)
            else:
                gen_deprecated_cc_list.append(op_def_cc)

        op_size = len(gen_include_list)
        max_op_size_in_one_file = 300
        save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_CC_PATH)
        for numbering in range(math.ceil(op_size / max_op_size_in_one_file)):
            gen_include = ''.join(
                gen_include_list[numbering * max_op_size_in_one_file: (numbering + 1) * max_op_size_in_one_file])
            gen_cc = ''.join(
                gen_cc_list[numbering * max_op_size_in_one_file: (numbering + 1) * max_op_size_in_one_file])
            cc_ops_def = self.CC_OPS_DEF_TEMPLATE.replace(gen_include=gen_include,
                                                          gen_cc_code=gen_cc)

            file_name = f"gen_ops_def_{chr(ord('a') + numbering)}.cc"
            ops_def_cc_file_str = template.CC_LICENSE_STR + cc_ops_def
            gen_utils.save_file(save_path, file_name, ops_def_cc_file_str)

        deprecated_cc_ops_def = self.CC_OPS_DEF_TEMPLATE.replace(gen_include='',
                                                                 gen_cc_code=''.join(gen_deprecated_cc_list))
        file_name = "gen_deprecated_ops_def.cc"
        deprecated_ops_def_cc_file_str = template.CC_LICENSE_STR + deprecated_cc_ops_def
        gen_utils.save_file(save_path, file_name,
                            deprecated_ops_def_cc_file_str)

    def process_args(self, op_args):
        """
        Processes operator arguments to extract input names.

        Args:
            op_args (list): A list of operator arguments.

        Returns:
            list: A list of input argument names.
        """
        inputs_name = []
        for arg in op_args:
            if not arg.is_prim_init:
                inputs_name.append(arg.arg_name)
        return inputs_name


class CustomOpsDefCcGenerator(OpsDefCcGenerator):
    """
    Generates C++ definition files for operators.
    """

    def __init__(self):
        """
        Initializes templates for generating C++ operator definitions.
        """
        super(CustomOpsDefCcGenerator, self).__init__()

        self.include_template = template.Template("""#include "${path}/${operator_name}.h\"\n""")
        self.func_impl_declaration_template = template.Template("extern OpFuncImpl &g${class_name}FuncImpl;")
        self.empty_func_impl_declaration_template = template.Template("static OpFuncImpl g${class_name}FuncImpl;")
        self.func_impl_define_template = template.Template("g${class_name}FuncImpl")
        self.OP_PROTO_TEMPLATE = template.OP_PROTO_TEMPLATE
        self.CC_OPS_DEF_TEMPLATE = template.Template(CC_OPS_DEF)

    def generate(self, work_path, op_protos):
        """
        Generates C++ code for operator definitions and saves it to a file.

        Args:
            work_path (str): The directory to save the generated files.
            op_protos (list): A list of operator prototypes.
        """
        gen_cc_list = list()
        gen_include_list = list()

        for op_proto in op_protos:
            class_name = "Custom_" + op_proto.op_name
            func_impl_declaration_str = self.func_impl_declaration_template.replace(class_name=class_name)
            func_impl_define = self.func_impl_define_template.replace(class_name=class_name)

            # process input
            args_dict, cc_index_str, input_args_str = process_input_args(op_proto)

            # Process outputs.
            return_args_str = get_cc_op_def_return(args_dict, op_proto)

            inputs_args = self.process_args(op_proto.op_args)
            signature_code = generate_cc_op_signature(op_proto.op_args_signature, inputs_args)
            enable_dispatch = "true" if op_proto.op_dispatch and op_proto.op_dispatch.enable else "false"
            is_view = "true" if op_proto.op_view else "false"
            is_graph_view = "true" if op_proto.op_graph_view else "false"
            op_def_cc = self.OP_PROTO_TEMPLATE.replace(class_name=class_name,
                                                       input_args=input_args_str,
                                                       return_args=return_args_str,
                                                       signatures=signature_code,
                                                       indexes=cc_index_str,
                                                       enable_dispatch=enable_dispatch,
                                                       is_view=is_view,
                                                       is_graph_view=is_graph_view,
                                                       func_impl_declaration=func_impl_declaration_str,
                                                       func_impl_define=func_impl_define)

            gen_cc_list.append(op_def_cc)

        gen_include = ''.join(gen_include_list)
        gen_cc = ''.join(gen_cc_list)
        cc_ops_def = self.CC_OPS_DEF_TEMPLATE.replace(
                                                      gen_include=gen_include,
                                                      gen_cc_code=gen_cc)

        file_name = f"gen_custom_ops_def.cc"
        ops_def_cc_file_str = template.CC_LICENSE_STR + cc_ops_def
        gen_utils.save_file(work_path, file_name, ops_def_cc_file_str)


def process_input_args(op_proto: OpProto):
    """
    Processes input arguments for C++ code generation.

    Args:
        op_proto (OpProto): The operator prototype.

    Returns:
        tuple: A tuple containing processed argument data.
    """
    cc_index_str = ''
    input_args_str = ''
    args_dict = {}
    op_args = op_proto.op_args
    for i, op_arg in enumerate(op_args):
        arg_name = op_arg.arg_name
        args_dict[arg_name] = i
        cc_index_str += f"""{{"{arg_name}", {i}}},\n"""
        dtype = op_arg.arg_dtype
        cc_dtype_str = gen_utils.convert_dtype_str(dtype)

        is_prim_init = 1 if op_arg.is_prim_init else 0
        arg_handler_str = op_arg.arg_handler

        type_cast = op_arg.type_cast
        type_cast_str = "" if type_cast is None else \
            ", ".join('DT_' + type.replace('[', '_').replace(']', '').upper() for type in type_cast)

        # default: None is regarded as an optional argument.
        is_optional_str = "true" if op_arg.default == "None" else "false"

        input_args_str += f"""\n    {{/*.arg_name_=*/"{arg_name}", /*.arg_dtype_=*/{cc_dtype_str}, """ + \
                          f"""/*.as_init_arg_=*/{is_prim_init}, /*.arg_handler_=*/"{arg_handler_str}", """ + \
                          f"""/*.cast_dtype_ =*/{{{type_cast_str}}}, /*.is_optional_=*/{is_optional_str}}},"""
    return args_dict, cc_index_str, input_args_str


def get_cc_op_def_return(args_dict, op_proto: OpProto):
    """
    Generates return argument strings for C++ operator definition.

    Args:
        args_dict (dict): A dictionary mapping argument names to indexes.
        op_proto (OpProto): The operator prototype.

    Returns:
        str: A string containing return argument data.
    """
    return_args_str = ''
    returns = op_proto.op_returns
    for return_item in returns:
        return_name = return_item.arg_name
        return_dtype = return_item.arg_dtype
        ref_name = return_item.inplace
        ref_index_str = args_dict.get(ref_name) if ref_name else -1
        cc_return_type_str = 'DT_' + return_dtype.replace('[', '_').replace(']', '').upper()
        return_args_str += f"""{{/*.arg_name_=*/"{return_name}", /*.arg_dtype_=*/{cc_return_type_str},
            /*.inplace_input_index_=*/{ref_index_str}}},\n"""
    return return_args_str


def generate_cc_op_signature(args_signature, args_name):
    """
    Generates C++ signature code for operator arguments.

    Args:
        args_signature (dict): A dictionary containing argument signatures.
        args_name (list): A list of argument names.

    Returns:
        str: A string containing the generated signature code.
    """
    if args_signature is None:
        return ''
    signature_code = ''

    # Init rw.
    read_list, ref_list, write_list = gen_utils.init_args_signature_rw(args_signature)

    # Init dtype group.
    same_dtype_groups, _ = gen_utils.get_same_dtype_groups(args_signature, args_name)
    for arg_name in args_name:
        enum_rw = signature_get_rw_label_cc(arg_name, write_list, read_list, ref_list)
        enum_dtype = signature_get_enum_dtype_cc(same_dtype_groups.get(arg_name))
        signature = f"""Signature("{arg_name}", {enum_rw}, """ \
                    f"""         SignatureEnumKind::kKindPositionalKeyword, nullptr, {enum_dtype}),\n """
        signature_code += signature
    return signature_code


def signature_get_rw_label_cc(rw_op_name, write_list, read_list, ref_list):
    """
    Determines the read-write label for a C++ signature.

    Args:
        rw_op_name (str): The name of the read-write operation.
        write_list (list): A list of write operations.
        read_list (list): A list of read operations.
        ref_list (list): A list of reference operations.

    Returns:
        str: The read-write label code.
    """
    # Define a dictionary mapping operation names to their corresponding RW labels
    rw_label_map = {
        'kRWWrite': write_list,
        'kRWRead': read_list,
        'kRWRef': ref_list
    }

    # Initialize with the default label
    rw_label = 'kRWDefault'

    # Check each list to see if the operation name matches and update the label if it does
    for label, names in rw_label_map.items():
        if rw_op_name in names:
            rw_label = label
            break  # Exit the loop once a match is found

    return f'SignatureEnumRW::{rw_label}'


def signature_get_enum_dtype_cc(index):
    """
    Generates C++ enum data type code for a signature.

    Args:
        index (int): The index of the data type.

    Returns:
        str: The enum data type code.
    """
    enum_type = 'SignatureEnumDType::'
    type_map = {0: 'kDType',
                1: 'kDType1',
                2: 'kDType2',
                3: 'kDType3',
                4: 'kDType4',
                5: 'kDType5',
                6: 'kDType6',
                7: 'kDType7',
                8: 'kDType8',
                9: 'kDType9'}
    if index in type_map:
        return enum_type + type_map[index]
    return enum_type + 'kDTypeEmptyDefaultValue'
