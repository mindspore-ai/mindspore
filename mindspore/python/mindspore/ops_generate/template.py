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
import re
import os


class CppTemplate:
    """
    template for generate c++ code
    """
    regular_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    regular_match = re.compile(regular_str, re.MULTILINE)

    def __init__(self, code_pattern):
        self.code_pattern = code_pattern

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, "r") as f:
            return CppTemplate(f.read())

    def replace(self, **kwargs):
        """
        replace param.
        :param kwargs:
        :return:
        """

        def find(key: str):
            if key in kwargs:
                return kwargs[key]
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
                    return add_indent(indent, [var])
                return add_indent(indent, var)
            if isinstance(var, list):
                code = ", ".join(str(x) for x in var)
                if not var:
                    return code
                return start + code + end
            return str(var)

        return self.regular_match.sub(match_rule, self.code_pattern)


NEW_LINE = "\n"
WORK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../")

PYTHON_PRIM_TEMPLATE = CppTemplate("""

class _Pyboost${class_name}Prim(${class_name}Prim_):
    def __call__(self, ${input_args}):
        ${process_func}
        return _convert_stub(super().__call__(${input_args}))


${func_impl_name}_impl = _Pyboost${class_name}Prim()
""")

IMPORT_PYBOOST_PRIM_HEADER = f"""
from mindspore.common._stub_tensor import _convert_stub
from mindspore.ops.auto_generate.gen_arg_handler import *
"""

IMPORT_PYBOOST_FUNC_HEADER = f"""
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate.pyboost_inner_prim import *

"""

REGISTER_DEFINE_TEMPLATE = CppTemplate(
    """
    (void)py::class_<${class_name}PrimAdapter, PrimitiveFunctionAdapter, std::shared_ptr<${class_name}PrimAdapter>>(
      *m, "${class_name}Prim_")
      .def(py::init<>())
      .def("__call__", &${class_name}PrimAdapter::Call, "Call ${class_name} op.");
    m->def(\"${pyboost_op_name}\", &mindspore::pynative::${pyboost_cfunc_name}, \"Encrypt the data.\");""")
REGISTER_TEMPLATE = CppTemplate("void RegisterPyBoostFunction(py::module *m) {${register_func}\n}")

REGISTER_PYBOOST_GRAD_DEFINE_TEMPLATE = CppTemplate(
    "MS_REG_PYBOOST_GRAD_OP(${pyboost_op_name}, mindspore::runtime::${pyboost_cfunc_name});\n")
REGISTER_PYBOOST_GRAD_TEMPLATE = CppTemplate("${register_func}")

PYBOOST_FUNCTION_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_function.tpl'))

PYBOOST_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_function_header.tpl'))

PYBOOST_GRAD_FUNCTION_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/runtime/pynative/op_function/template/pyboost_grad_function.tpl'))

PYBOOST_GRAD_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/runtime/pynative/op_function/template/pyboost_grad_function_header.tpl'))

GEN_OPS_DEF_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/python/mindspore/ops_generate/gen_ops_def_header.tpl'))

PYBOOST_BASE_OP_DEFINE_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/kernel/pyboost/template/pyboost_op_header.tpl'))

PYBOOST_OP_REGISTER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/kernel/pyboost/template/pyboost_op_register.tpl'))

# Ascend op generate
PYBOOST_ASCEND_OP_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/template/pyboost_aclnn_header_template.tpl'))

PYBOOST_ASCEND_OP_SOURCE_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/template/pyboost_aclnn_source_template.tpl'))

PYBOOST_ASCEND_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/template/pyboost_ascend_call_template.tpl'))

PYBOOST_ASCEND_VIEW_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/kernel/pyboost/template/'
                 'pyboost_view_template.tpl'))

PYBOOST_ASCEND_CUSTOMIZE_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/template'
                 '/pyboost_ascend_customize_call_template.tpl'))

# GPU op generate
PYBOOST_GPU_OP_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/gpu/kernel/pyboost/template/pyboost_gpu_header_template.tpl'))

PYBOOST_GPU_OP_SOURCE_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/gpu/kernel/pyboost/template/pyboost_gpu_source_template.tpl'))

PYBOOST_GPU_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/gpu/kernel/pyboost/template/pyboost_gpu_call_template.tpl'))

PYBOOST_GPU_VIEW_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/kernel/pyboost/template/pyboost_view_template.tpl'))

PYBOOST_GPU_CUSTOMIZE_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/gpu/kernel/pyboost/template'
                 '/pyboost_gpu_customize_call_template.tpl'))

# CPU op generate
PYBOOST_CPU_OP_HEADER_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/cpu/kernel/pyboost/template/pyboost_cpu_header_template.tpl'))

PYBOOST_CPU_OP_SOURCE_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/cpu/kernel/pyboost/template/pyboost_cpu_source_template.tpl'))

PYBOOST_CPU_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/cpu/kernel/pyboost/template/pyboost_cpu_call_template.tpl'))

PYBOOST_CPU_VIEW_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/kernel/pyboost/template/pyboost_view_template.tpl'))

PYBOOST_CPU_CUSTOMIZE_CALL_TEMPLATE = CppTemplate.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/plugin/device/cpu/kernel/pyboost/template'
                 '/pyboost_cpu_customize_call_template.tpl'))

PYBOOST_PY_FUNC_IMPORT_HEADEAR = CppTemplate(
    """from mindspore._c_expression import ${class_name}Prim_\n"""
)

PYBOOST_PY_FUNC_TEMPLATE = CppTemplate("""
def ${func_name}(${func_args}):
    r\"\"\"
    ${description}
    \"\"\"
    return ${func_impl_name}_impl(${input_args})\n\n""")

OP_PROTO_TEMPLATE = CppTemplate("""
${class_name}FuncImpl g${class_name}FuncImpl;
OpDef g${class_name} = {
  /*.name_=*/"${class_name}",
  /*.args_=*/ {
    ${input_args}
  },
  /* .returns_ = */ {
    ${return_args} 
  },
  /*.signatures_ =*/ {
    ${signatures}
  },
  /*.indexes_ =*/ {
    ${indexes}
  },
  /*.func_impl_=*/g${class_name}FuncImpl,
  /*.enable_dispatch_ =*/${enable_dispatch},
  /*.is_view_ =*/${is_view},
};
""")
