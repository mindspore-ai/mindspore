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
Module for generating Python primitive operator definitions from specifications.
"""

import os

import common.gen_constants as K
import common.template_utils as template
from common import gen_utils
from common.op_proto import OpProto
from common.template_utils import Template
from pyboost import pyboost_utils
from op_def_py.base_op_prim_py_generator import BaseOpPrimPyGenerator, _generate_arg_handler, generate_py_op_deprecated


class OpPrimPyGenerator(BaseOpPrimPyGenerator):
    """
    Generates Python code for primitive operators based on provided specifications.
    """

    def __init__(self):
        """
        Initializes the generator with a template for defining operator primitive classes.
        """
        self.op_prim_class_define_template = template.OP_PRIM_CLASS_DEFINE_TEMPLATE

    def generate(self, work_path, op_protos, doc_dict, file_pre):
        """
        Generates Python code for operator primitives and saves it to a file.

        Args:
            work_path (str): The directory to save the generated files.
            op_protos (list): A list of operator prototypes.
            doc_dict (dict): A dictionary containing documentation strings.
            file_pre (str): The prefix for the generated file names.
        """
        gen_py = ""
        for op_proto in op_protos:
            if op_proto.op_class.disable:
                continue

            inputs_args, inputs_default, init_args, args_assign, init_args_with_default, args_handlers = (
                self._process_args(op_proto))

            # add class description
            class_desc = self._generate_class_desc(op_proto, inputs_args, init_args, doc_dict)

            # add signature
            signature_code = self._generate_py_op_signature(op_proto, inputs_args, inputs_default)

            # add deprecated
            deprecated_code = generate_py_op_deprecated(op_proto.op_deprecated)

            # add __init__ method code
            init_method = self._generate_init_code(args_assign, init_args_with_default, op_proto)

            # add __call__ method code
            call_method = self._generate_call_code(args_handlers, init_args, inputs_args, inputs_default, op_proto)

            # generate op prim class define
            op_prim_class_define = self.op_prim_class_define_template.replace(class_name=op_proto.op_class.name,
                                                                              class_desc=class_desc,
                                                                              signature_code=signature_code,
                                                                              deprecated_code=deprecated_code,
                                                                              init_method=init_method,
                                                                              call_method=call_method)
            op_prim_class_define += "\n" if call_method.endswith("\n") else ""
            gen_py += op_prim_class_define

            # add prim_op_object
            if not init_args:
                gen_py += f"\n\n{op_proto.op_name}_op={op_proto.op_class.name}()\n"

        pyboost_import_header = self.generate_pyboost_import_header(op_protos)
        res_str = template.PY_LICENSE_STR + \
                  template.OPS_PY_PRIM_HEADER + pyboost_import_header + gen_py

        save_path = os.path.join(work_path, K.PY_AUTO_GEN_PATH)
        file_name = f"{file_pre}_ops_prim.py"
        gen_utils.save_file(save_path, file_name, res_str)

    def generate_pyboost_import_header(self, op_protos) -> str:
        """
        Generates import statements for PyBoost primitives.

        Args:
            op_protos (list): A list of operator prototypes.

        Returns:
            str: A string containing import statements.
        """
        pyboost_import_header = ''
        import_pyboost = Template("from mindspore._c_expression import $var\n")
        for op_proto in op_protos:
            if op_proto.op_dispatch and op_proto.op_dispatch.enable:
                header = import_pyboost.replace(var=pyboost_utils.get_pyboost_name(op_proto.op_name))
                pyboost_import_header += header
        return pyboost_import_header

    def _generate_init_code(self, args_assign, init_args_with_default, op_proto: OpProto):
        """
        Generates the __init__ method code for the operator primitive class.

        Args:
            args_assign (list): List of argument assignment strings.
            init_args_with_default (list): List of initialization arguments with default values.
            op_proto (OpProto): The operator prototype.

        Returns:
            str: A string containing the __init__ method code.
        """
        init_code_str = ""
        init_args_list_str = ""
        if init_args_with_default:
            init_args_list_str += ", " + f"""{", ".join(init_args_with_default) if init_args_with_default else ""}"""
        init_code = "\n".join(args_assign)
        init_code = self._get_init_code(init_code, op_proto)
        init_code_str += "    @prim_arg_register\n"
        init_code_str += f"    def __init__(self{init_args_list_str}):\n"
        init_code_str += f"{init_code}\n"
        init_code_str += "\n"
        return init_code_str

    def _get_call_method_body_str(self, args_handlers, init_args, inputs_args, inputs_default, op_proto: OpProto):
        """
        Generates the body of the __call__ method.

        Args:
            args_handlers (dict): Dictionary of argument handlers.
            init_args (list): List of initialization argument names.
            inputs_args (list): List of input argument names.
            inputs_default (dict): Dictionary of default input values.
            op_proto (OpProto): The operator prototype.

        Returns:
            str: A string containing the body of the call method.
        """
        call_args_list_str = ""
        replace_node_dict = {}
        pyboost_call_args_list_str = ""
        pyboost_arg_handlers_black_list = pyboost_utils.get_pyboost_arg_handlers_black_list()

        if inputs_args:
            args_with_handler = []
            pyboost_args_with_handler = []
            for arg in inputs_args:
                if arg in args_handlers:
                    is_optional = inputs_default.get(arg) == "None"
                    new_arg = _generate_arg_handler(
                        op_proto.op_class.name, arg, args_handlers[arg], is_optional)
                    replace_node_dict.update({arg: new_arg})
                    args_with_handler.append("new_generated_" + arg)
                else:
                    args_with_handler.append(arg)

                if arg in args_handlers and args_handlers[arg] not in pyboost_arg_handlers_black_list:
                    is_optional = inputs_default.get(arg) == "None"
                    new_arg = _generate_arg_handler(
                        op_proto.op_class.name, arg, args_handlers[arg], is_optional)
                    pyboost_args_with_handler.append(new_arg)
                else:
                    pyboost_args_with_handler.append(arg)

            call_args_list_str += ", ".join(args_with_handler)
            pyboost_call_args_list_str += ", ".join(pyboost_args_with_handler)

        if init_args:
            call_args_list_str += ", "
            call_args_list_str += ", ".join([f'self.{arg}' for arg in init_args])
            pyboost_call_args_list_str += ", "
            pyboost_call_args_list_str += ", ".join([f'self.{arg}' for arg in init_args])

        call_method_body_str = ""

        pyboost_call_method_body_str = ""
        is_pyboost = op_proto.op_dispatch and op_proto.op_dispatch.enable
        if is_pyboost:
            pyboost_func_name = pyboost_utils.get_pyboost_name(op_proto.op_name)
            pyboost_call_method_body_str = f"""
        res = {pyboost_func_name}(self, [{pyboost_call_args_list_str}])
        if not jit_context():
            return res"""
        call_method_body_str += pyboost_call_method_body_str

        node_pass = ""
        for name, inner in replace_node_dict.items():
            call_method_body_str += f"""
        new_generated_{name} = {inner}"""
            node_pass += f"""
        jit_context().pass_trace_node({name}, new_generated_{name})"""

        if is_pyboost:
            jit_context_call_method_body_str = """
        if jit_context().compiled:
            return jit_context().default_output()"""
            jit_context_call_method_body_str += node_pass
            jit_context_call_method_body_str += f"""
        return jit_context().run_op(self, res, {call_args_list_str})"""
            call_method_body_str += jit_context_call_method_body_str
        else:
            if node_pass != "":
                call_method_body_str += """
        if jit_context():"""
                node_pass = node_pass.replace('jit_context().pass_trace_node',
                                              '    jit_context().pass_trace_node')
            call_method_body_str += node_pass
            call_method_body_str += f"""
        return super().__call__({call_args_list_str})\n"""
        return call_method_body_str
