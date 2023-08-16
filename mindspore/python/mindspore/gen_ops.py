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
        class_def = operator_data.get('function')
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        if class_def:
            item = func_def.get("name")
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


def generate_py_primitive(yaml_data):
    """
    generate python primitive
    """
    gen_py = ''
    for operator_name, operator_data in yaml_data.items():
        args = operator_data.get('args')
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))

        init_args_with_default = []
        init_args = []
        args_assign = []
        for arg_name, arg_info in args.items():
            dtype = arg_info.get('dtype')
            type_cast = arg_info.get('type_cast')
            type_cast_set = None
            if type_cast:
                type_cast_set = {ct.strip() for ct in type_cast.split(",")}

            init_value = arg_info.get('init')
            if init_value is None:
                continue

            if dtype == 'str':
                init_value = '"' + init_value + '"'
            init_args_with_default.append(f"""{arg_name}={init_value}""")
            init_args.append(arg_name)

            assign_str = f"""        self.{arg_name} = """

            if type_cast_set:
                assign_str += f'type_it({arg_name}, '
                type_cast_list = []

                if 'int' in type_cast_set:
                    type_cast_list.append('INT')
                if 'tuple[int]' in type_cast_set:
                    type_cast_list.append('TUPLE')
                if 'scalar' in type_cast_set:
                    type_cast_list.append('SCALAR')
                #add more type cast kind here

                assign_str += 'TypeCastKind.' + '_OR_'.join(ct for ct in type_cast_list)
                if dtype == 'tuple[int]':
                    assign_str += '_TO_TUPLE)'
                if dtype == 'list[int]':
                    assign_str += '_TO_LIST)'
                if dtype == 'tensor':
                    assign_str += '_TO_TENSOR)'
            else:
                assign_str += arg_name
            args_assign.append(assign_str)

        args_assign = '\n'.join(assign for assign in args_assign)
        primitive_code = f"""
class {class_name}(Primitive):
    def __init__(self, {', '.join(init_args_with_default)}):
{args_assign}
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
    for operator_name, _ in yaml_data.items():
        OpName = ''.join(word.capitalize() for word in operator_name.split('_'))
        op_name_gen += f"""constexpr auto kName{OpName} = "{OpName}";
"""

    op_name_gen += op_name_end
    return op_name_gen


def generate_op_param_opdef(yaml_data):
    """
    generate BaseOperator parameter set and get func
    """
    op_param_head = f"""
#ifndef MINDSPORE_CORE_OP_PARAMETER_H_
#define MINDSPORE_CORE_OP_PARAMETER_H_

#include "ops/base_operator.h"
#include "ops/gen_ops_name.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {{
"""

    op_param_end = f"""}}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OP_PARAMETER_H_
"""

    op_param_gen = ''
    op_param_gen += op_param_head
    for operator_name, operator_data in yaml_data.items():
        OpName = ''.join(word.capitalize() for word in operator_name.split('_'))
        op_param_gen += f"""class MIND_API {OpName} : public BaseOperator {{
 public:
  {OpName}() : BaseOperator(kName{OpName}) {{}}
"""
        args = operator_data.get('args')
        for i, (arg_name, arg_info) in enumerate(args.items()):
            init = arg_info.get('init')
            if init is None:
                continue

            dtype = arg_info.get('dtype')
            if dtype == "str":
                dtype = "std::string"
            if dtype == "tuple[int]":
                dtype = "std::vector<int64_t>"
            op_param_gen += f"""  void set_{arg_name}(const {dtype} &{arg_name}) {{
    (void)this->AddAttr("{arg_name}", api::MakeValue({arg_name}));
  }}
"""
            op_param_gen += f"""  {dtype} get_{arg_name}() const {{
    return GetValue<{dtype}>(GetAttr("{arg_name}"));
  }}
"""

        op_param_gen += f"""}};
"""
    op_param_gen += op_param_end
    return op_param_gen


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
        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        opdef_map_str += f"""
    {{"{operator_name}", &g{class_name}}},"""
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
    work_path = ''
    if len(sys.argv) > 1:
        work_path = sys.argv[1]

    yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops.yaml')
    doc_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_doc.yaml')

    if len(sys.argv) > 3:
        yaml_path_root = sys.argv[2]
        op_name = sys.argv[3]
        yaml_path = os.path.join(work_path, f'{yaml_path_root}/{op_name}_op.yaml')
        doc_yaml_path = os.path.join(work_path, f'{yaml_path_root}/{op_name}_doc.yaml')

    op_py_path = os.path.join(work_path, 'mindspore/python/mindspore/gen_ops_def.py')
    op_cc_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_def.cc')
    op_name_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_name.h')
    op_param_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_param.h')

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

from mindspore.ops.primitive import Primitive
from mindspore.ops import operations as P
from mindspore.ops import functional as F
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

    op_param_code = generate_op_param_opdef(yaml_str)
    op_param_file = None
    with open(op_param_path, 'w') as op_param_file:
        op_param_file.write(cc_license_str + op_param_code)

    op_name_code = generate_op_name_opdef(yaml_str)
    op_name_file = None
    with open(op_name_path, 'w') as op_name_file:
        op_name_file.write(cc_license_str + op_name_code)
