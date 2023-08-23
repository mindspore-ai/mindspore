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
Generate operator definition from ops.yaml
"""
import sys
import os
import yaml

def generate_py_op_signature(args_signature):
    """
    generate __mindspore_signature__
    """
    if args_signature is None:
        return ''

    signature_code = f"""__mindspore_signature__ = """

    writable = args_signature.get('writable')
    same_type = args_signature.get('same_type')

    if writable is None:
        signature_code += '(sig.sig_dtype.T, sig.sig_dtype.T)'
        return signature_code

    # deal with writable
    writable = writable.replace(' ', '')
    same_type = same_type.replace(' ', '')

    signature_code += f""" (
"""
    same_type_list = []
    same_type_parsed = same_type.split("(")
    for item in same_type_parsed:
        if ')' in item:
            parsed = item.split(")")
            same_type_list.append(parsed[0])

    writable_items = writable.split(",")
    writable_items_used = [False for i in range(len(writable_items))]

    i = 0
    dtype = ''
    for same_type_team in same_type_list:
        if i == 0:
            dtype = f"""T"""
        else:
            dtype = f"""T{i}"""
        i = i + 1
        same_type_items = same_type_team.split(",")
        for same_type_i in same_type_items:
            find_writable = False
            for writable_index, item_name in enumerate(writable_items):
                if item_name == same_type_i:
                    find_writable = True
                    writable_items_used[writable_index] = True
                    signature_code += f"""     sig.make_sig('{item_name}', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.{dtype}),
"""
                    break
            if not find_writable:
                signature_code += f"""     sig.make_sig('{same_type_i}', dtype=sig.sig_dtype.{dtype}),
"""

    # item has writable but do not has same_type
    for used_index, used_item in enumerate(writable_items_used):
        if not used_item:
            item_name = writable_items[used_index]
            signature_code += f"""     sig.make_sig('{item_name}', sig.sig_rw.RW_WRITE),
"""

    signature_code += f"""    )"""

    return signature_code


def generate_py_op_func(yaml_data, doc_data):
    """
    generate python operator function
    """
    gen_py = ''

    op_desc_dict = {}
    for operator_name, operator_desc in doc_data.items():
        desc = operator_desc.get("description")
        op_desc_dict[operator_name] = desc

    for operator_name, operator_data in yaml_data.items():
        func_def = operator_data.get('function')
        func_name = operator_name
        func_disable = False
        if func_def:
            item = func_def.get("disable")
            if item:
                func_disable = True

            if func_disable:
                continue
            item = func_def.get("name")
            if item:
                func_name = item

        description = op_desc_dict.get(operator_name)
        args = operator_data.get('args')
        class_def = operator_data.get('class')
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        if class_def:
            item = class_def.get("name")
            if item:
                class_name = item
        func_args = []
        init_args = []
        input_args = []
        for arg_name, arg_info in args.items():
            dtype = arg_info.get('dtype')
            init_value = arg_info.get('init')
            if init_value is None:
                func_args.append(arg_name)
                input_args.append(arg_name)
            else:
                if dtype == 'str':
                    init_value = '"' + init_value + '"'
                func_args.append(f"""{arg_name}={init_value}""")
                init_args.append(arg_name)

        function_code = f"""
def {func_name}({', '.join(arg for arg in func_args)}):
    \"\"\"
    {description}
    \"\"\"
    {operator_name}_op = _get_cache_prim(P.{class_name})({', '.join(arg_name for arg_name in init_args)})
    return {operator_name}_op({', '.join(arg_name for arg_name in input_args)})
"""
        gen_py += function_code

    return gen_py


def process_args(args):
    """process arg for yaml, get arg_name, default value, cast type, pre-handler, etc."""
    args_name = []
    args_assign = []
    init_args_with_default = []
    for arg_name, arg_info in args.items():
        dtype = arg_info.get('dtype')

        init_value = arg_info.get('init')
        if init_value is None:
            continue
        if init_value == 'NO_VALUE':
            init_args_with_default.append(f"""{arg_name}""")
        elif init_value == 'None':
            init_args_with_default.append(f"""{arg_name}={init_value}""")
        else:
            if dtype == 'str':
                init_value = '"' + init_value + '"'
            init_args_with_default.append(f"""{arg_name}={init_value}""")
        args_name.append(arg_name)

        assign_str = ""
        type_cast = arg_info.get('type_cast')
        type_cast_set = None
        if type_cast:
            type_cast_set = {ct.strip() for ct in type_cast.split(",")}
        if type_cast_set:
            assign_str += f'type_it({arg_name}, '
            type_cast_list = []

            if 'int' in type_cast_set:
                type_cast_list.append('INT')
            if 'tuple[int]' in type_cast_set:
                type_cast_list.append('TUPLE')
            if 'scalar' in type_cast_set:
                type_cast_list.append('SCALAR')
            # add more type cast kind here

            assign_str += 'TypeCastKind.' + '_OR_'.join(ct for ct in type_cast_list)
            if dtype == 'tuple[int]':
                assign_str += '_TO_TUPLE)'
            if dtype == 'list[int]':
                assign_str += '_TO_LIST)'
            if dtype == 'tensor':
                assign_str += '_TO_TENSOR)'
        else:
            assign_str += arg_name

        arg_handler = arg_info.get('arg_handler')
        if arg_handler is not None:
            assign_str = f'arg_handle({assign_str}, ArgHandleKind.{arg_handler})'

        assign_str = f"""        self.{arg_name} = """ + assign_str
        args_assign.append(assign_str)
    return args_name, args_assign, init_args_with_default


def generate_py_primitive(yaml_data):
    """
    generate python primitive
    """
    gen_py = ''
    for operator_name, operator_data in yaml_data.items():
        signature_code = generate_py_op_signature(operator_data.get('args_signature'))

        args = operator_data.get('args')
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        class_def = operator_data.get('class')
        if class_def:
            item = class_def.get("name")
            if item:
                class_name = item

        init_args, args_assign, init_args_with_default = process_args(args)
        args_assign = '\n'.join(assign for assign in args_assign)
        primitive_code = f"""
class {class_name}(Primitive):
    {signature_code}
    @prim_attr_register
    def __init__(self, {', '.join(init_args_with_default) if init_args_with_default else ''}):
{args_assign if args_assign else '        pass'}

    def __call__(self, *args):
        super.__call__(self, *args, {', '.join([f'self.{arg}' for arg in init_args])})
"""

        gen_py += primitive_code
    return gen_py


def generate_op_name_opdef(yaml_data):
    """
    generate op name
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
        k_name_op = ''.join(word.capitalize() for word in operator_name.split('_'))
        class_def = operator_data.get('class')
        if class_def:
            item = class_def.get("name")
            if item:
                k_name_op = item
        op_name_gen += f"""constexpr auto kName{k_name_op} = "{k_name_op}";
"""

    op_name_gen += op_name_end
    return op_name_gen


def generate_lite_ops(yaml_data):
    """
    generate BaseOperator parameter set and get func
    """
    lite_ops_head = f"""
#ifndef MINDSPORE_CORE_LITE_OPS_H_
#define MINDSPORE_CORE_LITE_OPS_H_

#include "ops/base_operator.h"
#include "ops/gen_ops_name.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {{
"""

    lite_ops_end = f"""}}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_LITE_OPS_H_
"""

    lite_ops_gen = ''
    lite_ops_gen += lite_ops_head
    for operator_name, operator_data in yaml_data.items():
        OpName = ''.join(word.capitalize() for word in operator_name.split('_'))
        lite_ops_gen += f"""class MIND_API {OpName} : public BaseOperator {{
 public:
  {OpName}() : BaseOperator(kName{OpName}) {{}}
"""
        args = operator_data.get('args')
        for _, (arg_name, arg_info) in enumerate(args.items()):
            init = arg_info.get('init')
            if init is None:
                continue

            dtype = arg_info.get('dtype')
            if dtype == "str":
                dtype = "std::string"
            if dtype == "tuple[int]":
                dtype = "std::vector<int64_t>"
            lite_ops_gen += f"""  void set_{arg_name}(const {dtype} &{arg_name}) {{
    (void)this->AddAttr("{arg_name}", api::MakeValue({arg_name}));
  }}
"""
            lite_ops_gen += f"""  {dtype} get_{arg_name}() const {{
    return GetValue<{dtype}>(GetAttr("{arg_name}"));
  }}
"""

        lite_ops_gen += f"""}};

"""
    lite_ops_gen += lite_ops_end
    return lite_ops_gen


def generate_cc_opdef(yaml_data):
    """
    generate OpDef
    """
    func_suffix_str = 'FuncImpl'
    func_impl_dir = 'ops_func_impl'
    gen_cc = ''
    opdef_map_str = f"""
std::unordered_map<std::string, OpDefPtr> gOpDefTable = {{"""
    gen_include = ''

    for operator_name, operator_data in yaml_data.items():
        args = operator_data.get('args')
        returns = operator_data.get('returns')
        class_def = operator_data.get('class')
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        if class_def:
            item = class_def.get("name")
            if item:
                class_name = item
        opdef_map_str += f"""
    {{"{class_name}", &g{class_name}}},"""
        gen_include += f"""
#include "{func_impl_dir}/{operator_name}.h\""""

        opdef_cc = f"""
{class_name}{func_suffix_str} g{class_name}{func_suffix_str};"""
        opdef_cc += f"""
OpDef g{class_name} = {{
    .name_ = "{operator_name}","""
        opdef_cc += f"""
    .args_ = {{"""
        cc_index_str = f"""
    .indexes_ = {{"""

        for i, (arg_name, arg_info) in enumerate(args.items()):
            dtype = arg_info.get('dtype')
            init = arg_info.get('init')
            if init is None:
                init = 0
            else:
                init = 1
            cc_index_str += f"""
                {{"{arg_name}", {i}}},"""
            cc_dtype_str = 'DT_' + dtype.replace('[', '_').replace(']', '').replace('tuple', 'array').replace(
                'list', 'array').upper()
            cc_dtype_str.replace('TUPLE', 'ARRAY').replace('LIST', 'ARRAY')
            opdef_cc += f"""
                {{.arg_name_ = "{arg_name}", .arg_dtype_ = {cc_dtype_str}, .as_init_arg_ = {init}}},"""
        opdef_cc += f"""
    }},"""

        opdef_cc += f"""
    .returns_ = {{"""

        for return_name, return_info in returns.items():
            return_dtype = return_info.get('dtype')
            cc_return_type_str = 'DT_' + return_dtype.replace('[', '_').replace(']', '').replace(
                'tuple', 'array').replace('list', 'array').upper()
            opdef_cc += f"""
                {{.arg_name_ = "{return_name}", .arg_dtype_ = {cc_return_type_str}}},"""

        opdef_cc += f"""
    }},"""

        cc_index_str += f"""
    }},"""
        opdef_cc += cc_index_str

        cc_func_impl_str = f"""
    .func_impl_ = &g{class_name}{func_suffix_str},"""
        opdef_cc += cc_func_impl_str

        opdef_cc += f"""
}};"""

        gen_cc += opdef_cc

    opdef_map_str += f"""
}};"""
    gen_cc += opdef_map_str
    return gen_cc, gen_include


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    work_path = os.path.join(current_path, '../../../')
    if len(sys.argv) > 1:
        work_path = sys.argv[1]

    yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops.yaml')
    doc_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_doc.yaml')
    yaml_dir_path = os.path.join(work_path, 'mindspore/core/ops/ops_def/')


    if len(sys.argv) < 3:
        ops_yaml_str = 'echo "#gen ops yaml"> ' + f'{yaml_path}'
        os.system(ops_yaml_str)
        append_str = 'ls ' + f'{yaml_dir_path}' + '*op.yaml |xargs -i cat {} >> ' + f'{yaml_path}'
        os.system(append_str)

        doc_yaml_str = 'echo "#gen ops doc"> ' + f'{doc_yaml_path}'
        os.system(doc_yaml_str)
        doc_append_str = 'ls ' + f'{yaml_dir_path}' + '*doc.yaml |xargs -i cat {} >> ' + f'{doc_yaml_path}'
        os.system(doc_append_str)

    if len(sys.argv) > 3:
        yaml_path_root = sys.argv[2]
        op_name = sys.argv[3]
        yaml_path = os.path.join(work_path, f'{yaml_path_root}/{op_name}_op.yaml')
        doc_yaml_path = os.path.join(work_path, f'{yaml_path_root}/{op_name}_doc.yaml')

    op_py_path = os.path.join(work_path, 'mindspore/python/mindspore/gen_ops_def.py')
    op_cc_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_def.cc')
    op_name_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_name.h')
    lite_ops_path = os.path.join(work_path, 'mindspore/core/ops/gen_lite_ops.h')

    yaml_str = None
    with open(yaml_path, 'r') as yaml_file:
        yaml_str = yaml.safe_load(yaml_file)

    doc_str = None
    with open(doc_yaml_path, 'r') as doc_file:
        doc_str = yaml.safe_load(doc_file)

    cc_code, cc_include = generate_cc_opdef(yaml_str)
    cc_code += f"""
}}  // namespace mindspore::ops"""

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
    pyheader = f"""
\"\"\"Operators definition generated by gen_os.py, includes functions and primitive classes.\"\"\"

from mindspore.ops.primitive import Primitive, prim_attr_register
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import signature as sig
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.arg_dtype_cast import TypeCastKind, type_it
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

    ccheader = f"""
#include "op_def.h"
{cc_include}
namespace mindspore::ops {{
"""
    py_prim = generate_py_primitive(yaml_str)
    py_func = generate_py_op_func(yaml_str, doc_str)
    py_file = None
    with open(op_py_path, 'w') as py_file:
        py_file.write(py_licence_str + pyheader + py_prim + py_func)

    cc_file = None
    with open(op_cc_path, 'w') as cc_file:
        cc_file.write(cc_license_str + ccheader + cc_code)

    lite_ops_code = generate_lite_ops(yaml_str)
    lite_ops_file = None
    with open(lite_ops_path, 'w') as lite_ops_file:
        lite_ops_file.write(cc_license_str + lite_ops_code)

    op_name_code = generate_op_name_opdef(yaml_str)
    op_name_file = None
    with open(op_name_path, 'w') as op_name_file:
        op_name_file.write(cc_license_str + op_name_code)
