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
"""
Generate operator definition from ops.yaml
"""
import os
import re
import shutil
import pathlib
import logging
import gen_utils
from gen_utils import (py_licence_str, cc_license_str, check_change_and_replace_file, merge_files,
                       merge_files_append, safe_load_yaml, convert_dtype_str, write_file)
from pyboost_utils import get_pyboost_name, is_pyboost_enable, AclnnUtils, get_dtypes
import template
from template import CppTemplate
from gen_pyboost_func import gen_pyboost_code
from gen_aclnn_implement import gen_aclnn_kernel


def _get_op_name(yaml_key, yaml_value):
    """
    Get op name for python class Primitive or c++ OpDef name.
    """
    # If has class item, use the specified item.
    class_def = yaml_value.get("class")
    if class_def is not None:
        class_name_specify = class_def.get("name")
        if class_name_specify is not None:
            return class_name_specify
    # Else use the default rule generate class name.
    op_name = yaml_key
    class_name_normal = ''.join(word.capitalize() for word in op_name.split('_'))
    return class_name_normal


def _get_op_func_name(yaml_key, yaml_value):
    func_def = yaml_value.get('function')
    func_name = yaml_key

    if func_def is not None:
        item = func_def.get("name")
        if item is not None:
            func_name = item
    return func_name


def _auto_generate_class_disabled(yaml_value):
    """Check whether class can be auto generated."""
    if 'class' not in yaml_value.keys():
        return False
    class_def = yaml_value.get("class")
    if 'disable' not in class_def.keys():
        return False
    disable_item = class_def.get("disable")
    if disable_item is True:
        return True
    if disable_item is False:
        return False
    raise TypeError(f"The disable label for class should be True or False, but get {disable_item}.")


def _auto_generate_func_disabled(yaml_value):
    """Check whether function can be auto generated."""
    if 'function' not in yaml_value.keys():
        return False
    func_def = yaml_value.get('function')
    if 'disable' not in func_def.keys():
        return False
    disable_item = func_def.get("disable")
    if disable_item is True:
        return True
    if disable_item is False:
        return False
    raise TypeError(f"The disable label for function should be True or False, but get {disable_item}.")


def signature_get_rw_label(arg_name, write_list, read_list, ref_list):
    """
    Generate signature rw code
    """
    for rw_arg_name in write_list:
        if rw_arg_name == arg_name:
            return ', sig.sig_rw.RW_WRITE'
    for read_arg_name in read_list:
        if read_arg_name == arg_name:
            return ', sig.sig_rw.RW_READ'
    for ref_arg_name in ref_list:
        if ref_arg_name == arg_name:
            return ', sig.sig_rw.RW_REF'
    return ''


def signature_get_rw_label_cc(rw_op_name, write_list, read_list, ref_list):
    """
    Generate cc signature rw code
    """
    rw_label = 'kRWDefault'
    for op in write_list:
        if op == rw_op_name:
            rw_label = 'kRWWrite'
    for op in read_list:
        if op == rw_op_name:
            rw_label = 'kRWRead'
    for op in ref_list:
        if op == rw_op_name:
            rw_label = 'kRWRef'
    return 'SignatureEnumRW::' + rw_label


def signature_get_enum_dtype_cc(index):
    """
    Generate cc enum dtype code
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


def signature_get_dtype_label(index):
    """
    Generate signature dtype code
    """
    dtype_index = ''
    if index > 0:
        dtype_index = f"""{index}"""
    return f"""dtype=sig.sig_dtype.T{dtype_index}"""


def get_same_dtype_groups(args_signature, args_name):
    """
    Get same dtype groups
    """
    same_dtype_groups = {}
    dtype_conut = 0
    if args_signature is None:
        return same_dtype_groups, dtype_conut

    dtype_group = args_signature.get('dtype_group')
    if dtype_group is not None:
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
                        same_dtype_groups[item] = dtype_conut
                    break
            if not is_match:
                same_dtype_groups[arg_name] = dtype_conut
            dtype_conut = dtype_conut + 1
    return same_dtype_groups, dtype_conut


def generate_py_op_signature(op_name, args_signature, args_name, args_default):
    """
    Generate __mindspore_signature__
    """
    def _check_signature_arg_valid(op_name, sig_arg_names, args_names):
        for sig_arg_name in sig_arg_names:
            if sig_arg_name not in args_names:
                raise ValueError(f"Op {op_name} has no input arg named '{sig_arg_name}'!")

    if args_signature is None and not args_default:
        return ''

    signature_code = f"""    __mindspore_signature__ = """

    # Init rw.
    write_list = []
    read_list = []
    ref_list = []
    if args_signature is not None:
        rw_write = args_signature.get('rw_write')
        rw_read = args_signature.get('rw_read')
        rw_ref = args_signature.get('rw_ref')
        if rw_write is not None:
            write_list = rw_write.replace(' ', '').split(",")
            _check_signature_arg_valid(op_name, write_list, args_name)
        if rw_read is not None:
            read_list = rw_read.replace(' ', '').split(",")
            _check_signature_arg_valid(op_name, read_list, args_name)
        if rw_ref is not None:
            ref_list = rw_ref.replace(' ', '').split(",")
            _check_signature_arg_valid(op_name, ref_list, args_name)
    # Init dtype group.
    same_dtype_groups, dtype_conut = get_same_dtype_groups(args_signature, args_name)
    _check_signature_arg_valid(op_name, list(same_dtype_groups.keys()), args_name)
    # Only one dtype_group is set.
    if dtype_conut == 1 and not any([write_list, read_list, ref_list, args_default]):
        signature_code += '('
        for _ in range(len(args_name) - 1):
            signature_code += 'sig.sig_dtype.T, '
        signature_code += 'sig.sig_dtype.T)\n\n'
        return signature_code

    # Set sig.make_sig.
    signature_code += f""" (\n"""
    for arg_name in args_name:
        signature_code += f"""        sig.make_sig('{arg_name}'"""
        signature_code += signature_get_rw_label(arg_name, write_list, read_list, ref_list)
        if arg_name in same_dtype_groups:
            signature_code += f""", """ + signature_get_dtype_label(same_dtype_groups[arg_name])
        if arg_name in args_default:
            signature_code += f""", default=""" + str(args_default[arg_name])
        signature_code += f"""),\n"""
    signature_code += f"""    )\n\n"""
    return signature_code


def generate_cc_op_signature(args_signature, args_name):
    """
    generate signatures on in cc file
    :param args_signature:
    :param args_name:
    :return:
    """
    if args_signature is None:
        return ''
    signature_code = ''
    # Init rw.
    write_list = []
    read_list = []
    ref_list = []
    if args_signature is not None:
        rw_write = args_signature.get('rw_write')
        rw_read = args_signature.get('rw_read')
        rw_ref = args_signature.get('rw_ref')
        if rw_write is not None:
            write_list = rw_write.replace(' ', '').split(",")
        if rw_read is not None:
            read_list = rw_read.replace(' ', '').split(",")
        if rw_ref is not None:
            ref_list = rw_ref.replace(' ', '').split(",")
    # Init dtype group.
    same_dtype_groups, _ = get_same_dtype_groups(args_signature, args_name)
    for arg_name in args_name:
        enum_rw = signature_get_rw_label_cc(arg_name, write_list, read_list, ref_list)
        enum_dtype = signature_get_enum_dtype_cc(same_dtype_groups.get(arg_name))
        signature = f"""Signature("{arg_name}", {enum_rw}, \
         SignatureEnumKind::kKindPositionalKeyword, nullptr, {enum_dtype}),\n """
        signature_code += signature
    return signature_code


def generate_py_op_deprecated(deprecated):
    """
    Generate @deprecated
    """
    if deprecated is None:
        return ''
    version = deprecated.get("version")
    if version is None:
        raise ValueError("The version of deprecated can't be None.")
    substitute = deprecated.get("substitute")
    if substitute is None:
        raise ValueError("The substitute of deprecated can't be None.")
    use_substitute = deprecated.get("use_substitute")
    if use_substitute is None:
        raise ValueError("The use_substitute of deprecated can't be None.")
    if use_substitute is not True and use_substitute is not False:
        raise ValueError(f"The use_substitute must be True or False, but got {use_substitute}")

    deprecated = f"""    @deprecated("{version}", "{substitute}", {use_substitute})\n"""
    return deprecated


def _normalize_func_description_fromat(description):
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


def _get_op_description(operator_name, doc_str):
    """
    Generate ops api description.
    """
    if doc_str is None:
        print(f"Description is None, op_name: {operator_name}")
        return ""
    description = doc_str.get(operator_name)
    if description is None:
        print(f"Description is None, op_name: {operator_name}")
        return ""
    description = description.get("description")
    if description is None:
        print(f"Description is None, op_name: {operator_name}")
        return ""
    return _normalize_func_description_fromat(description)


def generate_py_op_func(yaml_data, doc_data):
    """
    Generate operator python function api.
    """
    gen_py = ''

    for operator_name, operator_data in yaml_data.items():
        if _auto_generate_func_disabled(operator_data):
            continue
        func_name = _get_op_func_name(operator_name, operator_data)
        args = operator_data.get('args')
        class_name = _get_op_name(operator_name, operator_data)
        func_args = []
        prim_init_args = []
        prim_call_args = []
        for arg_name, arg_info in args.items():
            is_prim_init = arg_info.get('prim_init')
            has_default = 'default' in arg_info.keys()

            # step1: Process function args.
            if not has_default:
                func_args.append(f"""{arg_name}""")
            else:
                default_value = arg_info.get('default')
                func_args.append(f"""{arg_name}={default_value}""")

            # step2: Process primitive object init args.
            if is_prim_init:
                prim_init_args.append(arg_name)

            # step3: Process primitive object call args.
            else:
                prim_call_args.append(arg_name)
        description = _get_op_description(operator_name, doc_data)
        function_code = f"""\n
def {func_name}({', '.join(arg for arg in func_args)}):
    r\"\"\"
    {description}
    \"\"\"
    {operator_name}_op = _get_cache_prim({class_name})({', '.join(arg_name for arg_name in prim_init_args)})
    return {operator_name}_op({', '.join(arg_name for arg_name in prim_call_args)})\n"""

        if not prim_init_args:
            if _auto_generate_class_disabled(operator_data):
                gen_py += f"""\n{operator_name}_op={class_name}()"""
            function_code = f"""\n
def {func_name}({', '.join(arg for arg in func_args)}):
    r\"\"\"
    {description}
    \"\"\"
    return {operator_name}_op({', '.join(arg_name for arg_name in prim_call_args)})\n"""
        else:
            dis = operator_data.get("dispatch")
            if dis is not None:
                enable_pyboost = dis.get("enable")
                if enable_pyboost:
                    function_code = f"""\n
def {func_name}({', '.join(arg for arg in func_args)}):
    r\"\"\"
    {description}
    \"\"\"
    return {operator_name}_impl({', '.join(arg_name for arg_name, _ in args.items())})\n"""
        gen_py += function_code

    return gen_py


def get_dtype(arg_info):
    dtype = arg_info.get('dtype')
    # Currently, TypeId is represented by int
    if dtype == 'TypeId':
        dtype = 'int'
    return dtype


def process_args(class_name, args):
    """
    Process arg for yaml, get arg_name, init value, type cast, arg_handler, etc.
    """
    inputs_name = []
    args_name = []
    args_assign = []
    inputs_default = {}
    init_args_with_default = []
    args_handlers = {}
    for arg_name, arg_info in args.items():
        dtype = get_dtype(arg_info)
        default_value = arg_info.get('default')
        has_default = 'default' in arg_info.keys()
        is_prim_init = arg_info.get('prim_init')
        arg_handler = arg_info.get('arg_handler')

        # step1: get args infos:
        if is_prim_init:
            # step1.1: get args name:
            args_name.append(arg_name)
            # step1.2: get args assign with default value:
            if has_default:
                init_args_with_default.append(f"""{arg_name}={default_value}""")
            else:
                init_args_with_default.append(f"""{arg_name}""")

            # step1.3: get args set prim arg expression:
            assign_str = gen_utils.get_assign_str_by_type_it(class_name, arg_info, arg_name, dtype)
            if arg_handler:
                assign_str = f"""        self._set_prim_arg_with_handler("{arg_name}", {assign_str}, {arg_handler})"""
            else:
                assign_str = f"""        self._set_prim_arg("{arg_name}", {assign_str})"""
            args_assign.append(assign_str)
        # step2: get inputs infos:
        else:
            # step2.1: get inputs name:
            inputs_name.append(arg_name)

            # step2.2: get default value of inputs:
            if has_default:
                inputs_default[arg_name] = default_value

            # step2.3: get args_handler functions for inputs
            if arg_handler:
                args_handlers[arg_name] = arg_handler

    return inputs_name, inputs_default, args_name, args_assign, init_args_with_default, args_handlers


def generate_pyboost_import_header(yaml_data):
    """
    Generate python primitive
    """
    pyboost_import_header = ''
    import_pyboost = CppTemplate("from mindspore._c_expression import $var\n")
    for operator_name, operator_data in yaml_data.items():
        is_pyboost = is_pyboost_enable(operator_data)
        if is_pyboost:
            header = import_pyboost.replace(var=get_pyboost_name(operator_name))
            pyboost_import_header += header
    return pyboost_import_header


def _generate_class_description(class_name, func_name, input_args, init_args, func_disabled, doc_str):
    """Generate description for every primitive definition."""
    if func_disabled:
        # if function disabled, function name is equal to operator_name
        description = _get_op_description(func_name, doc_str)
        description = f"""    r\"\"\"
    {description}
    \"\"\"
"""
        return description

    # If function is an released API, refer to the function doc.
    description_str = f"""    r\"\"\"
    .. code-block::
        
        prim = ops.{class_name}({', '.join(init_args)})
        out = prim({', '.join(input_args)})
        
    is equivalent to
    
    .. code-block::
    
        ops.{func_name}({", ".join(input_args + init_args)})
        
    Refer to :func:`mindspore.ops.{func_name}` for more details.
    \"\"\"
"""
    return description_str


def get_init_code(init_code, operator_data):
    """
    Generate init code for primitive
    """
    labels = operator_data.get('labels')
    if labels is not None:
        if init_code != "":
            init_code += "\n"
        init_code += \
            '\n'.join([f"""        self.add_prim_attr("{key}", {value})""" for key, value in labels.items()])
    if init_code == "":
        init_code = f"""        pass"""
    return init_code


def generate_py_primitive(yaml_data, doc_str):
    """
    Generate python primitive
    """

    def _generate_arg_handler(class_name, arg, arg_handler, is_optional):
        """Generate arg_handler"""
        arg_handler_call = f"""{arg_handler}('{class_name}', '{arg}', {arg})"""
        if is_optional:
            arg_handler_call = f"""{arg} if {arg} is None else {arg_handler_call}"""
        return arg_handler_call

    gen_py = ''
    for operator_name, operator_data in yaml_data.items():
        if _auto_generate_class_disabled(operator_data):
            continue
        class_name = _get_op_name(operator_name, operator_data)
        func_name = _get_op_func_name(operator_name, operator_data)
        pyboost_func_name = get_pyboost_name(operator_name)
        args = operator_data.get('args')
        inputs_args, inputs_default, init_args, args_assign, init_args_with_default, args_handlers = \
            process_args(class_name, args)
        init_code = '\n'.join(args_assign)
        signature_code = generate_py_op_signature(class_name, operator_data.get('args_signature'), inputs_args,
                                                  inputs_default)
        deprecated_code = generate_py_op_deprecated(operator_data.get('deprecated'))
        init_code = get_init_code(init_code, operator_data)
        primitive_code = f"""\n
class {class_name}(Primitive):\n"""
        func_disabled = _auto_generate_func_disabled(operator_data)
        primitive_code += _generate_class_description(class_name, func_name, inputs_args, init_args, func_disabled,
                                                      doc_str)
        if signature_code != "":
            primitive_code += signature_code
        if deprecated_code != "":
            primitive_code += deprecated_code
        primitive_code += f"""    @prim_arg_register
    def __init__(self"""
        if init_args_with_default:
            primitive_code += ", " + f"""{', '.join(init_args_with_default) if init_args_with_default else ''}"""
        call_args = []
        for name in inputs_args:
            call_args.append(f"""{name}={inputs_default[name]}""" if name in inputs_default else name)
        primitive_code += f"""):
{init_code}

    def __call__(self, {', '.join(call_args)}):"""
        is_pyboost = is_pyboost_enable(operator_data)
        if is_pyboost:
            primitive_code += f"""
          return _convert_stub({pyboost_func_name}(self, ["""
        else:
            primitive_code += f"""
          return super().__call__("""
        if inputs_args:
            args_with_handler = []
            for arg in inputs_args:
                if arg in args_handlers:
                    is_optional = inputs_default.get(arg) == "None"
                    args_with_handler.append(_generate_arg_handler(class_name, arg, args_handlers[arg], is_optional))
                else:
                    args_with_handler.append(arg)
            primitive_code += ', '.join(args_with_handler)

        if init_args:
            primitive_code += ', '
            primitive_code += ', '.join([f'self.{arg}' for arg in init_args])
        if is_pyboost:
            primitive_code += """]))"""
        else:
            primitive_code += """)
"""

        gen_py += primitive_code
        if not init_args:
            prim_op_object = f"""\n
{operator_name}_op={class_name}()
"""
            gen_py += prim_op_object
    return gen_py


def generate_op_name_opdef(yaml_data):
    """
    Generate op name
    """
    op_name_head = f"""
#ifndef MINDSPORE_CORE_OP_NAME_H_
#define MINDSPORE_CORE_OP_NAME_H_

namespace mindspore::ops {{
"""

    op_name_end = f"""}}  // namespace mindspore::ops

#endif  // MINDSPORE_CORE_OP_NAME_H_
"""

    op_name_gen = ''
    op_name_gen += op_name_head
    for operator_name, operator_data in yaml_data.items():
        k_name_op = _get_op_name(operator_name, operator_data)
        op_name_gen += f"""constexpr auto kName{k_name_op} = "{k_name_op}";
"""

    op_name_gen += op_name_end
    return op_name_gen


def generate_op_prim_opdef(yaml_data):
    """
    Generate primitive c++ definition
    """
    ops_prim_head = f"""
#ifndef MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_
#define MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "mindapi/base/macros.h"

namespace mindspore::prim {{
"""

    ops_prim_end = f"""}}  // namespace mindspore::prim
#endif  // MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_
"""

    ops_prim_gen = ''
    ops_prim_gen += ops_prim_head
    for operator_name, operator_data in yaml_data.items():
        k_name_op = _get_op_name(operator_name, operator_data)
        ops_prim_gen += f"""GVAR_DEF(PrimitivePtr, kPrim{k_name_op}, std::make_shared<Primitive>(ops::kName{k_name_op}))
"""
    ops_prim_gen += ops_prim_end
    return ops_prim_gen


def generate_lite_ops(yaml_data):
    """
    Generate BaseOperator parameter set and get func
    """
    lite_ops_h_head = f"""
#ifndef MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_
#define MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_

#include <vector>
#include "ops/base_operator.h"
#include "ops/auto_generate/gen_ops_name.h"

namespace mindspore::ops {{
"""

    lite_ops_h_end = f"""}}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_
"""

    lite_ops_cc_head = """
#include "ops/auto_generate/gen_lite_ops.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
"""

    lite_ops_cc_end = f"""}}  // namespace mindspore::ops
    """

    lite_ops_h_gen = ''
    lite_ops_cc_gen = ''

    lite_ops_h_gen += lite_ops_h_head
    lite_ops_cc_gen += lite_ops_cc_head
    for operator_name, operator_data in yaml_data.items():
        op_name = _get_op_name(operator_name, operator_data)
        lite_ops_h_gen += f"""class MIND_API {op_name} : public BaseOperator {{
 public:
  MIND_API_BASE_MEMBER({op_name});
  {op_name}() : BaseOperator(kName{op_name}) {{}}\n"""
        args = operator_data.get('args')
        for _, (arg_name, arg_info) in enumerate(args.items()):
            is_prim_init = arg_info.get('prim_init')
            if not is_prim_init:
                continue

            dtype = get_dtype(arg_info)
            if dtype == "str":
                dtype = "std::string"
            if dtype in ("tuple[str]", "list[str]"):
                dtype = "std::vector<std::string>"
            if dtype in ("tuple[int]", "list[int]"):
                dtype = "std::vector<int64_t>"
            if dtype in ("tuple[float]", "list[float]"):
                dtype = "std::vector<float>"
            if dtype in ("tuple[bool]", "list[bool]"):
                dtype = "std::vector<bool>"
            if dtype == "int":
                dtype = "int64_t"
            lite_ops_h_gen += f"""  void set_{arg_name}(const {dtype} &{arg_name});\n"""
            lite_ops_h_gen += f"""  {dtype} get_{arg_name}() const;\n"""

            lite_ops_cc_gen += f"""void {op_name}::set_{arg_name}(const {dtype} &{arg_name}) \
            {{ (void)this->AddAttr("{arg_name}", api::MakeValue({arg_name})); }}\n\n"""
            lite_ops_cc_gen += f"""{dtype} {op_name}::get_{arg_name}() const \
            {{ return GetValue<{dtype}>(GetAttr("{arg_name}")); }}\n\n"""

            op_name = _get_op_name(operator_name, operator_data)
        lite_ops_cc_gen += f"""REGISTER_PRIMITIVE_C(kName{op_name}, {op_name});\n"""
        lite_ops_cc_gen += f"""MIND_API_OPERATOR_IMPL({op_name}, BaseOperator);\n\n"""
        lite_ops_h_gen += f"""}};\n\n"""
    lite_ops_h_gen += lite_ops_h_end
    lite_ops_cc_gen += lite_ops_cc_end
    return lite_ops_h_gen, lite_ops_cc_gen


def generate_cc_opdef(yaml_data):
    """
    Generate c++ OpDef
    """
    gen_cc_code = f"""\n
namespace mindspore::ops {{"""
    gen_opdef_map = f"""
std::unordered_map<std::string, OpDefPtr> gOpDefTable = {{"""
    gen_include = f"""\n
#include \"ops/auto_generate/gen_ops_def.h\""""
    gen_include += f"""
#include \"mindspore/core/ir/signature.h\""""

    for operator_name, operator_data in yaml_data.items():
        args = operator_data.get('args')
        class_name = _get_op_name(operator_name, operator_data)
        inputs_args, _, _, _, _, _ = process_args(class_name, args)
        signature_code = generate_cc_op_signature(operator_data.get('args_signature'), inputs_args)
        args = operator_data.get('args')
        returns = operator_data.get('returns')
        dispatch = operator_data.get("dispatch")
        # dispatch not defined in yaml or dispatch.enable==False
        if not dispatch or not dispatch.get("enable"):
            dispatch = "false"
        else:
            dispatch = "true"
        enable_dispatch_str = f"""{dispatch}"""

        is_view = operator_data.get('view')
        if is_view:
            is_view = "true"
        else:
            is_view = "false"
        is_view_str = f"""{is_view}"""


        gen_include += f"""\n#include "ops/ops_func_impl/{operator_name}.h\""""
        cc_index_str = ''
        gen_opdef_map += f"""\n  {{"{class_name}", &g{class_name}}},"""
        input_args_str = ''
        args_dict = {}
        for i, (arg_name, arg_info) in enumerate(args.items()):
            args_dict[arg_name] = i
            cc_index_str += f"""{{"{arg_name}", {i}}},\n"""
            dtype = get_dtype(arg_info)
            cc_dtype_str = convert_dtype_str(dtype)

            is_prim_init = 1 if arg_info.get('prim_init') else 0
            arg_handler = arg_info.get('arg_handler')
            arg_handler_str = "" if arg_handler is None else arg_handler

            type_cast = arg_info.get('type_cast')
            type_cast_str = "" if type_cast is None else \
                ', '.join('DT_' + type.replace('[', '_').replace(']', '').upper() for type in
                          (ct.strip() for ct in type_cast.split(",")))

            # default: None is regarded as a optional argument.
            is_optional_str = "false"
            if 'default' in arg_info.keys() and arg_info.get('default') == "None":
                is_optional_str = "true"

            input_args_str += f"""\n    {{/*.arg_name_=*/"{arg_name}", /*.arg_dtype_=*/{cc_dtype_str}, """ + \
                              f"""/*.as_init_arg_=*/{is_prim_init}, /*.arg_handler_=*/"{arg_handler_str}", """ + \
                              f"""/*.cast_dtype_ =*/{{{type_cast_str}}}, /*.is_optional_=*/{is_optional_str}}},"""

        # Process outputs.
        return_args_str = ''
        for return_name, return_info in returns.items():
            return_dtype = return_info.get('dtype')
            ref_name = return_info.get('inplace')
            ref_index_str = -1 if ref_name is None else args_dict.get(ref_name)
            cc_return_type_str = 'DT_' + return_dtype.replace('[', '_').replace(']', '').upper()
            return_args_str += f"""{{/*.arg_name_=*/"{return_name}", /*.arg_dtype_=*/{cc_return_type_str},
            /*.inplace_input_index_=*/{ref_index_str}}},\n"""

        op_def_cc = template.OP_PROTO_TEMPLATE.replace(class_name=class_name, input_args=input_args_str,
                                                       return_args=return_args_str, signatures=signature_code,
                                                       indexes=cc_index_str, enable_dispatch=enable_dispatch_str,
                                                       is_view=is_view_str)
        gen_cc_code += op_def_cc
    gen_opdef_map += f"""\n}};"""
    gen_cc_code += gen_opdef_map

    cc_opdef_end = f"""\n}}  // namespace mindspore::ops\n"""
    return gen_include + gen_cc_code + cc_opdef_end


ops_py_prim_header = f"""
\"\"\"Operators definition generated by gen_ops.py, includes primitive classes.\"\"\"

from mindspore.ops.primitive import Primitive, prim_arg_register
from mindspore.ops import signature as sig
from mindspore.common import dtype as mstype
from mindspore.common._decorator import deprecated
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_arg_dtype_cast import type_it
from mindspore.ops.auto_generate.gen_arg_handler import *
from mindspore._c_expression import OpDtype
from mindspore.common._stub_tensor import _convert_stub
"""


ops_py_def_header = f"""
\"\"\"Operators definition generated by gen_ops.py, includes functions.\"\"\"

from .gen_ops_prim import *
from .pyboost_inner_prim import *
from mindspore.ops.operations.manually_defined.ops_def import *
from mindspore.ops._primitive_cache import _get_cache_prim
"""


def generate_ops_prim_file(work_path, yaml_str, doc_str, file_pre):
    py_path = os.path.join(work_path, f'mindspore/python/mindspore/ops/auto_generate/{file_pre}_ops_prim.py')
    tmp_py_path = os.path.join(work_path, f'mindspore/python/mindspore/ops/auto_generate/tmp_{file_pre}_ops_prim.py')
    pyboost_import_header = generate_pyboost_import_header(yaml_str)
    py_prim = generate_py_primitive(yaml_str, doc_str)
    write_file(tmp_py_path, py_licence_str + ops_py_prim_header + pyboost_import_header + py_prim)
    check_change_and_replace_file(py_path, tmp_py_path)


def generate_ops_def_file(work_path, yaml_str, doc_str, file_pre):
    py_path = os.path.join(work_path, f'mindspore/python/mindspore/ops/auto_generate/{file_pre}_ops_def.py')
    tmp_py_path = os.path.join(work_path, f'mindspore/python/mindspore/ops/auto_generate/tmp_{file_pre}_ops_def.py')
    py_func = generate_py_op_func(yaml_str, doc_str)
    write_file(tmp_py_path, py_licence_str + ops_py_def_header + py_func)
    check_change_and_replace_file(py_path, tmp_py_path)


def generate_ops_py_files(work_path, yaml_str, doc_str, file_pre):
    """
    Generate ops python file from yaml.
    """
    generate_ops_prim_file(work_path, yaml_str, doc_str, file_pre)
    generate_ops_def_file(work_path, yaml_str, doc_str, file_pre)


def generate_ops_cc_files(work_path, yaml_str):
    """
    Generate ops c++ file from yaml.
    """
    # ops_def
    op_cc_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/gen_ops_def.cc')
    tmp_op_cc_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/tmp_gen_ops_def.cc')
    cc_def_code = generate_cc_opdef(yaml_str)
    write_file(tmp_op_cc_path, cc_license_str + cc_def_code)
    check_change_and_replace_file(op_cc_path, tmp_op_cc_path)

    # ops_primitive
    op_prim_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/gen_ops_primitive.h')
    tmp_op_prim_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/tmp_gen_ops_primitive.h')
    op_prim_code = generate_op_prim_opdef(yaml_str)
    write_file(tmp_op_prim_path, cc_license_str + op_prim_code)
    check_change_and_replace_file(op_prim_path, tmp_op_prim_path)

    # lite_h_ops
    lite_ops_h_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/gen_lite_ops.h')
    tmp_lite_ops_h_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/tmp_gen_lite_ops.h')
    lite_ops_h_code, lite_ops_cc_code = generate_lite_ops(yaml_str)
    write_file(tmp_lite_ops_h_path, cc_license_str + lite_ops_h_code)
    check_change_and_replace_file(lite_ops_h_path, tmp_lite_ops_h_path)

    # lite_cc_ops
    lite_ops_cc_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/gen_lite_ops.cc')
    tmp_lite_ops_cc_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/tmp_gen_lite_ops.cc')
    write_file(tmp_lite_ops_cc_path, cc_license_str + lite_ops_cc_code)
    check_change_and_replace_file(lite_ops_cc_path, tmp_lite_ops_cc_path)

    # ops_names
    op_name_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/gen_ops_name.h')
    tmp_op_name_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/tmp_gen_ops_name.h')
    op_name_code = generate_op_name_opdef(yaml_str)
    write_file(tmp_op_name_path, cc_license_str + op_name_code)
    check_change_and_replace_file(op_name_path, tmp_op_name_path)


def generate_op_labels(yaml_data):
    """
    Generate python labels
    """
    gen_label_py = f"""op_labels = {{"""
    for operator_name, operator_data in yaml_data.items():
        labels = operator_data.get('labels')
        if labels is not None:
            class_name = _get_op_name(operator_name, operator_data)
            gen_label_py += f"""
    "{class_name}": {{"""
            gen_label_py += f""", """.join([f""""{key}": {value}""" for key, value in labels.items()])
            gen_label_py += f"""}},"""
    gen_label_py += f"""
}}"""
    return gen_label_py


def generate_op_arg_default_value(yaml_data):
    """
    Generate python default value.
    """
    default_py_header = f"""\"\"\"Operator labels and args default value.\"\"\"
from mindspore.common import dtype as mstype\n\n"""

    gen_default_py = default_py_header + f"""op_args_default_value = {{"""
    for operator_name, operator_data in yaml_data.items():
        arg_default_dict = {}
        args = operator_data.get('args')
        for arg_name, arg_info in args.items():
            arg_default = arg_info.get('default')
            if arg_default is not None:
                arg_default_dict[arg_name] = arg_default
        if arg_default_dict:
            class_name = _get_op_name(operator_name, operator_data)
            gen_default_py += f"""
    "{class_name}": {{"""
            gen_default_py += f""", """.join([f""""{key}": {value}""" for key, value in arg_default_dict.items()])
            gen_default_py += f"""}},"""
    gen_default_py += f"""
}}"""
    return gen_default_py


def generate_create_instance_helper_file(work_path, yaml_str):
    """
    Generate C++ helper file from yaml.
    """
    dst_dir = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate')
    op_py_path = os.path.join(dst_dir, 'cpp_create_prim_instance_helper.py')
    tmp_op_py_path = os.path.join(dst_dir, 'tmp_cpp_create_prim_instance_helper.py')
    py_labels = generate_op_labels(yaml_str)
    py_arg_default = generate_op_arg_default_value(yaml_str)
    write_file(tmp_op_py_path, py_licence_str + "\n" + py_arg_default + "\n\n" + py_labels + "\n")
    check_change_and_replace_file(op_py_path, tmp_op_py_path)


def generate_aclnn_reg_code(yaml_data):
    """generate aclnn register code"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    work_path = os.path.join(current_path, '../../../../')
    ops_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops.yaml')
    yaml_str = gen_utils.safe_load_yaml(ops_yaml_path)

    reg_code = f"""
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"

namespace mindspore {{
namespace kernel {{
"""
    for operator_name, operator_data in yaml_data.items():
        dispatch = operator_data.get("dispatch")
        if not dispatch or not dispatch.get("enable"):
            continue
        Ascend = dispatch.get("Ascend")
        if Ascend is not None:  # KernelMod is provided by yaml, don't auto generate it.
            continue
        _, _, none_tensor_exist = get_dtypes(operator_data)
        if none_tensor_exist:
            gen_aclnn_kernel(operator_name, yaml_str, auto=True)
            continue
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        op_class = operator_data.get("class")
        if op_class and op_class.get("name") is not None:
            class_name = op_class.get("name")
        inputs_outputs_num = len(operator_data.get("args")) + len(operator_data.get("returns"))
        aclnn_name = AclnnUtils.get_aclnn_interface(class_name)
        reg_code += f"""
MS_ACLNN_COMMON_KERNEL_FACTORY_REG({class_name}, {aclnn_name}, {inputs_outputs_num});"""
    reg_code += f"""
}}  // namespace kernel
}}  // namespace mindspore
"""
    return reg_code


def generate_aclnn_reg_file(work_path, yaml_str):
    """
    Generate nnacl kernelmod register
    """
    tmp_register_file = work_path + 'mindspore/ccsrc/plugin/device/ascend/kernel/opapi/tmp_aclnn_kernel_register.cc'
    register_file = work_path + 'mindspore/ccsrc/plugin/device/ascend/kernel/opapi/aclnn_kernel_register_auto.cc'
    reg_code = generate_aclnn_reg_code(yaml_str)
    write_file(tmp_register_file, cc_license_str + reg_code)
    check_change_and_replace_file(register_file, tmp_register_file)


def generate_arg_handler_files(work_path):
    """
    Generate arg handler files.
    """
    dst_dir = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate')
    src_arg_handler_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/arg_handler.py')
    dst_arg_handler_path = os.path.join(dst_dir, 'gen_arg_handler.py')
    tmp_dst_arg_handler_path = os.path.join(dst_dir, 'tmp_gen_arg_handler.py')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(src_arg_handler_path, tmp_dst_arg_handler_path)
    check_change_and_replace_file(dst_arg_handler_path, tmp_dst_arg_handler_path)

    src_arg_dtype_cast_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/arg_dtype_cast.py')
    dst_arg_dtype_cast_path = os.path.join(dst_dir, 'gen_arg_dtype_cast.py')
    tmp_arg_dtype_cast_path = os.path.join(dst_dir, 'tmp_arg_dtype_cast.py')
    shutil.copy(src_arg_dtype_cast_path, tmp_arg_dtype_cast_path)
    check_change_and_replace_file(dst_arg_dtype_cast_path, tmp_arg_dtype_cast_path)


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    work_path = os.path.join(current_path, '../../../../')

    # merge ops yaml
    ops_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops.yaml')
    doc_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops_doc.yaml')

    ops_yaml_dir_path = os.path.join(work_path, 'mindspore/core/ops/ops_def/')
    infer_ops_yaml_dir_path = os.path.join(work_path, 'mindspore/core/ops/ops_def/infer/')
    doc_yaml_dir_path = os.path.join(work_path, 'mindspore/core/ops/ops_def/doc/')
    merge_files(ops_yaml_dir_path, ops_yaml_path, '*op.yaml')
    merge_files_append(infer_ops_yaml_dir_path, ops_yaml_path, '*op.yaml')
    merge_files(doc_yaml_dir_path, doc_yaml_path, '*doc.yaml')

    # make auto_generate dir
    cc_path = os.path.join(work_path, 'mindspore/core/ops/auto_generate/')
    pathlib.Path(cc_path).mkdir(parents=True, exist_ok=True)

    # generate arg_handler files
    generate_arg_handler_files(work_path)

    # read ops definition str and doc str
    ops_yaml_str = safe_load_yaml(ops_yaml_path)
    doc_yaml_str = safe_load_yaml(doc_yaml_path)

    # generate ops python files
    generate_ops_py_files(work_path, ops_yaml_str, doc_yaml_str, "gen")

    # generate ops c++ files
    generate_ops_cc_files(work_path, ops_yaml_str)
    # generate create prim instance helper file
    generate_create_instance_helper_file(work_path, ops_yaml_str)
    # generate pyboost code
    gen_pyboost_code(work_path, ops_yaml_str, doc_yaml_str)
    # generate aclnn kernelmod register
    generate_aclnn_reg_file(work_path, ops_yaml_str)


if __name__ == "__main__":
    try:
        main()
    # pylint: disable=broad-except
    except Exception as e:
        logging.critical("Auto generate failed, err info: %s", e)
