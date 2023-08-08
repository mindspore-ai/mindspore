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
        description = op_desc_dict.get(operator_name)
        args = operator_data.get('args')
        func_name = operator_data.get('func_name')
        if func_name is None:
            func_name = operator_name

        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        func_args = []
        primitive_init_args = []
        input_args = []
        for arg_name, arg_info in args.items():
            dtype = arg_info.get('dtype')
            init_value = arg_info.get('init')
            if init_value:
                if dtype == 'str':
                    init_value = '"' + init_value + '"'
                func_args.append(f"""{arg_name}={init_value}""")
                primitive_init_args.append(arg_name)
            else:
                func_args.append(arg_name)
                input_args.append(arg_name)

        function_code = f"""
def {func_name}({', '.join(arg for arg in func_args)}):
    \"\"\"
    {description}
    \"\"\"
    {operator_name}_op = _get_cache_prim(P.{class_name})({', '.join(arg_name for arg_name in primitive_init_args)})
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
        func_name = operator_data.get('func_name')
        if func_name is None:
            func_name = operator_name

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
                if 'tuple[int]' in type_cast_list:
                    type_cast_list.append('TUPLE')
                #add more type cast kind here

                assign_str += 'TypeCastKind.' + '_OR_'.join(ct for ct in type_cast_list)
                if dtype == 'tuple[int]':
                    assign_str += '_TO_TUPLE)'
                if dtype == 'list[int]':
                    assign_str += '_TO_LIST)'
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


def generate_cc_opdef(yaml_data):
    """
    generate OpDef
    """
    gen_cc = ''
    opdef_map_str = f"""
std::unordered_map<std::string, OpDefPtr> gOpDefTable = {{"""

    for operator_name, operator_data in yaml_data.items():
        args = operator_data.get('args')
        returns = operator_data.get('returns')
        func_name = operator_data.get('func_name')
        if func_name is None:
            func_name = operator_name

        class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
        opdef_map_str += f"""
    {{"{operator_name}", &g{class_name}}},"""

        opdef_cc = f"""
OpDef g{class_name} = {{
    .name_ = "{operator_name}","""
        opdef_cc += f"""
    .args_ = {{"""

        for arg_name, arg_info in args.items():
            dtype = arg_info.get('dtype')
            init = arg_info.get('init')
            if init is None:
                init = 0
            else:
                init = 1
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

        opdef_cc += f"""
}};"""
        gen_cc += opdef_cc

    opdef_map_str += f"""
}};"""
    gen_cc += opdef_map_str
    return gen_cc


if __name__ == "__main__":
    work_path = ''
    if len(sys.argv) > 1:
        work_path = sys.argv[1]

    yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops.yaml')
    doc_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_doc.yaml')
    op_py_path = os.path.join(work_path, 'mindspore/python/mindspore/gen_ops_def.py')
    op_cc_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_def.cc')

    yaml_str = None
    with open(yaml_path, 'r') as yaml_file:
        yaml_str = yaml.safe_load(yaml_file)

    doc_str = None
    with open(doc_yaml_path, 'r') as doc_file:
        doc_str = yaml.safe_load(doc_file)

    cc_code = generate_cc_opdef(yaml_str)
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
