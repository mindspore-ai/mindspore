import re
import os

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
"""Template."""


class CppTemplate:
    regular_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    regular_match = re.compile(regular_str, re.MULTILINE)

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, "r") as f:
            return CppTemplate(f.read())

    def __init__(self, code_pattern):
        self.code_pattern = code_pattern

    def replace(self, **kwargs):

        def find(key: str):
            if key in kwargs:
                return kwargs[key]
            else:
                raise TypeError(f"{key} should be in kwargs!")

        def add_indent(indent, var):
            return "".join([indent + line + "\n" for data in var for line in str(data).splitlines()]).rstrip()

        def extract_variable(key):
            start = ""
            end = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    start = ","
                    key = key[1:]
                if key[-1] == ",":
                    end = ", "
                    key = key[:-1]
            return find(key), start, end

        def match_rule(match):
            indent = match.group(1)
            key = match.group(2)
            var, start, end = extract_variable(key)

            if indent is not None:
                if not isinstance(var, list):
                    var = [var]
                return add_indent(indent, var)

            elif isinstance(var, list):
                code = ", ".join(str(x) for x in var)
                if len(var) == 0:
                    return code
                return start + code + end
            else:
                return str(var)

        return self.regular_match.sub(match_rule, self.code_pattern)


NEW_LINE = "\n"
WORK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../")

REGISTER_DEFINE_TEMPLATE = CppTemplate(
    "  m->def(\"${pyboost_op_name}\", &mindspore::pynative::${pyboost_cfunc_name}, \"Encrypt the data.\");\n")
REGISTER_TEMPLATE = CppTemplate("void RegisterPyBoostFunction(py::module *m) {\n${register_func}\n}\n")

PYBOOST_FUNCTION_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_function.tpl'))

PYBOOST_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_function_header.tpl'))

GEN_OPS_DEF_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/python/mindspore/ops_generate/gen_ops_def_header.tpl'))

PYBOOST_BASE_OP_DEFINE_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/kernel/pyboost/pyboost_op_header.tpl'))

PYBOOST_OP_REGISTER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/kernel/pyboost/pyboost_op_register.tpl'))

PYBOOST_ASCEND_OP_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/pyboost_aclnn_header_template.tpl'))

PYBOOST_ASCEND_OP_SOURCE_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/pyboost_aclnn_source_template.tpl'))

PYBOOST_ASCEND_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/pyboost_ascend_call_template.tpl'))

PYBOOST_VIEW_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/pyboost_view_call_template.tpl'))

PYBOOST_CUSTOMIZE_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/pyboost_ascend_customize_call_template.tpl'))

PYBOOST_PY_FUNC_HEADEAR = ("""
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_def import *
""")

PYBOOST_PY_FUNC_TEMPLATE = CppTemplate("""
def ${func_name}(${func_args}):
    r\"\"\"
    ${description}
    \"\"\"
    ${operator_name}_op = _get_cache_prim(${class_name})(${init_args})
    return ${operator_name}_op(${input_args})\n\n""")

