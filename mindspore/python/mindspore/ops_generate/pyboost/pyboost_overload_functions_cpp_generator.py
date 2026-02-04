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
This module defines the PyboostFunctionsGenerator class for generating C++ functions for PyBoost operations.

The generator processes operator prototypes and constructs the necessary function definitions, including
conversions for optional parameters and tensor arguments. It generates the registration code and includes
the necessary header files for the generated functions.
"""

import os

from common import template
import common.gen_constants as K
from common.template import Template
from common.gen_utils import save_file
from common.op_proto import OpProto
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils
from api import op_api_proto

from .op_template_parser import OpTemplateParser


class PyboostOverloadFunctionsGenerator(BaseGenerator):
    """
    Generates PyBoost overload functions cpp code based on operator prototypes.

    This class processes operator prototypes (`op_protos`) to create the necessary C++ function definitions for
    PyBoost operations. It constructs function bodies, handles optional value conversions, and generates
    registration code and header inclusions.
    """

    def __init__(self):
        self.PYBOOST_OVERLOAD_FUNCTIONS_TEMPLATE = template.PYBOOST_OVERLOAD_FUNCTIONS_CC_TEMPLATE
        self.PYBOOST_MINT_CLASS_DEF = template.PYBOOST_MINT_CLASS_DEF
        self.PYBOOST_OVERLOAD_MINT_CLASS_DEF = template.PYBOOST_OVERLOAD_MINT_CLASS_DEF
        self.TENSOR_FUNC_UT_BODY = template.TENSOR_FUNC_UT_BODY
        self.PYBOOST_OVERLOAD_UT_BODY = template.PYBOOST_OVERLOAD_UT_BODY

        self.single_case_template = Template(
            'case ${case_id}:\n'
            '  {\n'
            '  ${dispatch_lambda_def}\n'
            '  ${device_dispatcher}\n'
            '  }\n'
            '  break;\n'
        )
        self.device_dispatcher_template = Template(
            'if (backend == device::DeviceType::kAscend) {\n'
            '  ${ascend_dispatcher}\n'
            '} else if (backend == device::DeviceType::kCPU) {\n'
            '  ${cpu_dispatcher}\n'
            '} else if (backend == device::DeviceType::kGPU) {\n'
            '  ${gpu_dispatcher}\n'
            '} else {\n'
            '  MS_LOG(ERROR) << "Device target is not supported!";\n'
            '  return py::none();\n'
            '}'
        )
        self.pyboost_return_template = Template(
            'return pyboost_call();\n'
        )
        self.pyboost_call_lambda_template = Template(
            'const auto pyboost_call = [&]{\n'
            '  if (parse_args.has_fallback()) {\n'
            '    auto op_call = std::make_shared<OpCall>("${class_name}", callback);\n'
            '    return pynative::HandleFallback(args, kwargs, py::cast(op_call));\n'
            '  }\n'
            '  ${arg_handler_processor}\n'
            '  MS_LOG(INFO) << "Call Tensor${class_name}";\n'
            '  auto res = ${pyboost_base_func_name}_OP(${prim_name}, parse_args.src_types_, ${convert_args});\n'
            '  trace::CapturePy(parse_args.arg_list_, mindspore::prim::kPrim${class_name}, &res);\n'
            '  return py::reinterpret_steal<py::object>(res);\n'
            '};\n'
        )
        self.callback_python_lambda_template = Template(
            'const auto callback_python = [&]{\n'
            '  if (parse_args.has_fallback()) {\n'
            '    auto op_call = std::make_shared<OpCall>("${class_name}", callback);\n'
            '    return pynative::HandleFallback(args, kwargs, py::cast(op_call));\n'
            '  }\n'
            '  MS_LOG(INFO) << "Callback python method: ${py_method}";\n'
            '  py::function fn = python_adapter::GetPyFn(\"mindspore.ops.tensor_method\", \"${py_method}\");\n'
            '  py::object res = fn(*args, **kwargs);\n'
            '  return res;\n'
            '};\n'
        )
        self.callback_python_template = Template(
            'if (ops::IsOpPluginKernel("${op_name}")) {\n'
            '  return pyboost_call();\n'
            '}\n'
            'return callback_python();\n'
        )
        self.pybind_register_template = Template(
            '(void)py::class_<${cpp_func_name}Functional, Functional, std::shared_ptr<${cpp_func_name}Functional>>\n'
            '  (*m, "${cpp_func_name}Functional_")\n'
            '  .def("__call__", &${cpp_func_name}Functional::Call, "Call ${cpp_func_name} functional.");\n'
            'm->attr("_${mint_func_name}_instance") = ${mint_func_name}_instance;'
        )
        self.callback_python_in_ut_template = Template(
            'MS_LOG(INFO) << "Callback python method in UT: ${py_method}";\n'
            'fn = python_adapter::GetPyFn(\"mindspore.ops.tensor_method\", \"${py_method}\");\n'
            'res = fn(*args, **kwargs);\n'
            'break;\n'
        )
        self.single_case_in_ut_template = Template(
            'case ${case_id}:\n'
            '  ${device_dispatcher}\n'
        )

    def generate(self, work_path, op_protos, mint_func_protos_data, alias_func_mapping):
        """
        Generates the C++ PyBoost functions and writes them to the specified files.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information
        such as operator names, arguments, and conversion types. It constructs the function definitions, includes,
        and registration code. The generated content is saved to the specified path as a C++ source file.

        Args:
            work_path (str): The file path where the generated files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.
            mint_func_protos_data (dict): A dict of tensor prototypes containing device-related information.
            alias_func_mapping (dict): A dict mapping from api name to its alias api name.

        Returns:
            None
        """

        mint_classes_def_list = []
        ops_inc_head_set = set()
        _, single_mint_func_data, overload_mint_func_data, op_class_name_set = op_api_proto.categorize_func_data(
            mint_func_protos_data)
        single_func_call_body_list, single_cpp_class_name_list = (
            self._get_single_func_call_body_list(single_mint_func_data))
        overload_func_call_body_list, overload_cpp_class_name_list = (
            self._get_overload_func_call_body_list(overload_mint_func_data))

        mint_classes_def_list.extend(single_func_call_body_list)
        mint_classes_def_list.extend(overload_func_call_body_list)

        cpp_class_name_list = single_cpp_class_name_list + overload_cpp_class_name_list
        mint_classes_reg_list = (
            self._get_mint_func_reg_list(single_mint_func_data, overload_mint_func_data, cpp_class_name_list))
        for op_class_name in op_class_name_set:
            ops_inc_head_set.add(template.OP_DEF_INC_HEAD_TEMPLATE.replace(prefix_char=op_class_name[0].lower()))
        pyboost_overload_file_str = (
            self.PYBOOST_OVERLOAD_FUNCTIONS_TEMPLATE.replace(ops_inc=list(sorted(ops_inc_head_set)),
                                                             mint_func_classes_def=mint_classes_def_list,
                                                             pybind_register_code=mint_classes_reg_list))
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_overload_functions.cc"
        save_file(save_path, file_name, pyboost_overload_file_str)

    def _get_single_func_call_body_list(self, single_op_func_data):
        """
        Generates the list of call body strings for single operation functions.

        Args:
            single_op_func_data (dict): Dictionary of tensor function prototypes with only one definition.

        Returns:
            func_call_body_list (list): Updated str list for generating C++ function call bodies.
            cpp_class_name_list (list): The list of non-overloaded c++ functional classes' names.
        """
        func_call_body_list, cpp_class_name_list = [], []
        for _, func_proto in single_op_func_data.items():
            func_name = func_proto.func_name
            class_name = func_proto.op_proto.op_class.name
            dispatch_lambda_def_str = self._get_dispatch_lambda_def_str(func_proto)
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            # Combine dispatch_lambda_def with device_dispatcher for single case
            device_dispatcher_str = dispatch_lambda_def_str + device_dispatcher_str
            signature_str = self._generate_single_signature_str(
                func_proto.op_proto, func_proto.kw_only_args,
                func_proto.varargs, func_proto.disable_scalar_tensor
            )
            op_args = func_proto.op_proto.op_args
            max_size = len(op_args)
            ut_body = self.TENSOR_FUNC_UT_BODY.replace(py_method=func_proto.py_method)
            func_call_body = self.PYBOOST_MINT_CLASS_DEF.replace(
                class_name=class_name,
                func_name=func_name,
                device_dispatcher=device_dispatcher_str,
                signatures=signature_str,
                max_args=max_size,
                ut_body=ut_body)
            func_call_body_list.append(func_call_body)
            cpp_class_name_list.append(class_name)
        return func_call_body_list, cpp_class_name_list

    def _get_overload_func_call_body_list(self, overload_op_func_data):
        """
        Generates the list of call body strings for overloaded operation functions.

        Args:
            overload_op_func_data (dict): Dictionary of tensor function prototypes with overloaded definitions.

        Returns:
            func_call_body_list (list): Updated str list for generating C++ function call bodies.
            cpp_class_name_list (list): The list of overloaded c++ functional classes' names.
        """
        func_call_body_list, cpp_class_name_list = [], []
        for func_api_name, func_protos in overload_op_func_data.items():
            func_call_body_list.append(
                self._get_overload_func_call_str(func_api_name, func_protos, cpp_class_name_list))
        return func_call_body_list, cpp_class_name_list

    def _get_overload_func_call_str(self, func_api_name, func_protos, cpp_class_name_list):
        """
        Generates C++ call body string for overloaded tensor functions.

        Args:
            func_api_name (str): Name of the function API.
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated call body string for the overloaded functions.
        """
        signatures_str = self._generate_func_signatures_str(func_protos)
        dispatch_cases = self._get_dispatch_cases(func_protos)
        ut_dispatch_cases = self._get_ut_dispatch_cases(func_protos)
        ut_overload_body = self.PYBOOST_OVERLOAD_UT_BODY.replace(ut_dispatch_cases=ut_dispatch_cases)

        max_size = 0
        for tensor_proto in func_protos:
            op_proto = tensor_proto.op_proto
            op_args = op_proto.op_args
            max_size = max(len(op_args), max_size)
        cpp_func_name = pyboost_utils.format_func_api_name(func_api_name)
        cpp_class_name_list.append(cpp_func_name)
        overload_func_call_str = self.PYBOOST_OVERLOAD_MINT_CLASS_DEF.replace(cpp_func_name=cpp_func_name,
                                                                              func_name=func_api_name,
                                                                              signatures=signatures_str,
                                                                              dispatch_cases=dispatch_cases,
                                                                              max_args=max_size,
                                                                              ut_overload_body=ut_overload_body)
        return overload_func_call_str

    def _generate_func_signatures_str(self, func_protos) -> str:
        """
        Generates function signatures as a string from the given prototypes.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated function signatures string.
        """
        sig_str = ''
        first_sig = True
        for tensor_proto in func_protos:
            op_proto = tensor_proto.op_proto
            if not first_sig:
                sig_str += ',\n'
            first_sig = False
            sig_str += self._generate_single_signature_str(op_proto, tensor_proto.kw_only_args,
                                                           tensor_proto.varargs, tensor_proto.disable_scalar_tensor)
        return sig_str

    def _generate_single_signature_str(self, op_proto: OpProto, kw_only_args,
                                       varargs, disable_scalar_tensor) -> str:
        op_parser = OpTemplateParser(op_proto)
        return op_parser.generate_signature_str(kw_only_args, varargs,
                                                disable_scalar_tensor, is_tensor_api=False)

    def _get_dispatch_cases(self, func_protos):
        """
        Generates C++ switch-case statements for dispatching tensor function calls.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated switch-case dispatch statements.
        """
        dispatch_cases_str = ''
        for idx, func_proto in enumerate(func_protos):
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            dispatch_lambda_def_str = self._get_dispatch_lambda_def_str(func_proto)
            dispatch_cases_str += self.single_case_template.replace(case_id=idx,
                                                                    dispatch_lambda_def=dispatch_lambda_def_str,
                                                                    device_dispatcher=device_dispatcher_str)
        dispatch_cases_str += 'default:\n'
        dispatch_cases_str += '  return py::none();'
        return dispatch_cases_str

    def _get_ut_dispatch_cases(self, func_protos):
        """
        Generates C++ switch-case statements for dispatching tensor function calls.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated switch-case dispatch statements.
        """
        dispatch_cases_str = ''
        for idx, func_proto in enumerate(func_protos):
            device_dispatcher_str = self.callback_python_in_ut_template.replace(py_method=func_proto.py_method)
            dispatch_cases_str += self.single_case_in_ut_template.replace(case_id=idx,
                                                                          device_dispatcher=device_dispatcher_str)
        dispatch_cases_str += 'default:\n'
        dispatch_cases_str += '  res = py::none();'
        return dispatch_cases_str

    def _get_device_dispatchers_str(self, func_proto):
        """
        Generates device-specific dispatch strings for the given function prototype.

        Args:
            func_proto (TensorFuncProto): Function prototype to generate dispatch strings for.

        Returns:
            str: Generated device-specific dispatch string.
        """
        ascend_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'ascend')
        cpu_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'cpu')
        gpu_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'gpu')
        device_dispatcher_str = self.device_dispatcher_template.replace(ascend_dispatcher=ascend_dispatcher_str,
                                                                        cpu_dispatcher=cpu_dispatcher_str,
                                                                        gpu_dispatcher=gpu_dispatcher_str)
        return device_dispatcher_str

    def _get_single_device_dispatcher_str(self, func_proto, device):
        """
        Generates the dispatch string for a specific device.

        Args:
            func_proto (TensorFuncProto): Function prototype to generate the dispatcher for.
            device (str): Device type ('ascend', 'cpu', 'gpu').

        Returns:
            str: Generated device dispatcher string.
        """
        func_proto_device = getattr(func_proto, device)
        if func_proto_device == 'pyboost':
            arg_handler_processor_str = self._get_arg_handler_processor(func_proto.func_name, func_proto.op_proto)
            convert_args_str = self._get_convert_args_str(func_proto.op_proto)
            op_parser = OpTemplateParser(func_proto.op_proto)
            op_pyboost_func_name = op_parser.get_pyboost_func_name()
            prim_name = f"prim::kPrim{func_proto.op_proto.op_class.name}"
            return self.pyboost_return_template.replace(arg_handler_processor=arg_handler_processor_str,
                                                        class_name=func_proto.op_proto.op_class.name,
                                                        prim_name=prim_name,
                                                        op_name=func_proto.op_proto.op_name,
                                                        pyboost_base_func_name=op_pyboost_func_name,
                                                        convert_args=convert_args_str)
        if func_proto_device == 'py_method':
            return self.callback_python_template.replace(op_name=func_proto.op_proto.op_class.name)

        raise TypeError("Only support pyboost or python_method.")

    def _get_dispatch_lambda_def_str(self, func_proto):
        """
        Generates the dispatch lambda function definition for pyboost call and python callback

        Args:
            func_proto (TensorFuncProto): Function prototype to generate the dispatcher for.

        Returns:
            str: Generated dispatch lambda function definition string.
        """
        devices = ['cpu', 'gpu', 'ascend']
        arg_handler_processor_str = self._get_arg_handler_processor(func_proto.func_name, func_proto.op_proto)
        op_parser = OpTemplateParser(func_proto.op_proto)
        op_pyboost_func_name = op_parser.get_pyboost_func_name()
        convert_args_str = self._get_convert_args_str(func_proto.op_proto)
        prim_name = f"prim::kPrim{func_proto.op_proto.op_class.name}"

        dispatch_lambda_str = ''
        has_pyboost_call = False
        for device in devices:
            if getattr(func_proto, device) == 'pyboost':
                has_pyboost_call = True
                break
        if has_pyboost_call:
            dispatch_lambda_str += self.pyboost_call_lambda_template.replace(
                arg_handler_processor=arg_handler_processor_str,
                class_name=func_proto.op_proto.op_class.name, prim_name=prim_name,
                op_name=func_proto.op_proto.op_name,
                pyboost_base_func_name=op_pyboost_func_name,
                convert_args=convert_args_str)
        else:
            dispatch_lambda_str += 'const auto pyboost_call = []{ Py_RETURN_NONE; };\n'

        for device in devices:
            if getattr(func_proto, device) == 'py_method':
                dispatch_lambda_str += self.callback_python_lambda_template.replace(py_method=func_proto.py_method,
                                                            class_name=func_proto.op_proto.op_class.name)
                break

        return dispatch_lambda_str

    def _get_arg_handler_processor(self, func_name, op_proto):
        op_parser = OpTemplateParser(op_proto)
        return op_parser.get_arg_handler_processor(func_name, op_proto, is_tensor_api=False)

    def _get_convert_args_str(self, op_proto):
        op_parser = OpTemplateParser(op_proto)
        return op_parser.get_convert_args_str(op_proto, is_tensor_api=False)

    def _get_mint_func_reg_list(self, single_mint_func_data, overload_mint_func_data, cpp_class_names):
        """
        Generates the list of pybind definition strings for mint functions.

        Args:
            single_mint_func_data (dict): Dictionary of single mint function data.
            overload_mint_func_data (dict): Dictionary of overload mint function data.
            cpp_class_names (list): List of C++ class names.

        Returns:
            list: list of strs for generating pybind definitions of mint functions' API.
        """
        # the order of single_mint_func_data/overload_mint_func_data matters
        mint_func_names = list(single_mint_func_data.keys()) + list(overload_mint_func_data.keys())

        mint_func_reg_list = []
        for mint_func_name, cpp_func_name in zip(mint_func_names, cpp_class_names):
            mint_func_reg_list.append(self.pybind_register_template.replace(mint_func_name=mint_func_name,
                                                                            cpp_func_name=cpp_func_name))
        return mint_func_reg_list
