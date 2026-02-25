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
Generate pyboost function from pyboost_op.yaml
"""

from resources.resource_list import ResourceType
from common import gen_constants as K
from api.functions_cc_generator import FunctionsGenerator, FunctionsHeaderGenerator
from .pyboost_inner_prim_generator import PyboostInnerPrimGenerator
from .pyboost_functions_py_generator import PyboostFunctionsPyGenerator
from .pyboost_functions_h_generator import PyboostFunctionsHeaderGenerator
from .pyboost_functions_cpp_generator import PyboostFunctionsGenerator
from .pyboost_functions_impl_cpp_generator import PyboostFunctionsImplGenerator
from .pyboost_grad_function_cpp_generator import PyboostGradFunctionsGenerator
from .pyboost_internal_functions_h_generator import PyboostInternalFunctionsHeaderGenerator
from .pyboost_internal_functions_cpp_generator import PyboostInternalFunctionsCppGenerator
from .pyboost_internal_kernel_info_adapter_generator import PyboostKernelInfoAdapterGenerator
from .pyboost_native_grad_functions_generator import (
    PyboostGradFunctionsHeaderGenerator,
    PyboostGradFunctionsCppGenerator,
)
from .pyboost_op_cpp_code_generator import (
    PyboostCommonOpHeaderGenerator,
    PyboostOpFunctionGenerator,
    PyboostOpHeaderGenerator,
    PyboostInternalOpHeaderGenerator,
    delete_residual_files,
    PyboostOpRegisterCppCodeGenerator,
)
from .pyboost_overload_functions_cpp_generator import PyboostOverloadFunctionsGenerator
from .auto_grad_impl_cc_generator import AutoGradImplGenerator
from .auto_grad_reg_cc_generator import AutoGradRegHeaderGenerator


def gen_pyboost_code(resource_mgr):
    """ gen_pyboost_code """
    work_path = K.WORK_DIR
    op_protos = resource_mgr.get_resource(ResourceType.OP_PROTO)
    doc_yaml_data = resource_mgr.get_resource(ResourceType.OP_DOC_YAML)
    mint_func_protos = resource_mgr.get_resource(ResourceType.MINT_FUNC_PROTOS)
    alias_func_mapping = resource_mgr.get_resource(ResourceType.ALIAS_API_MAPPING)
    call_pyboost_inner_prim_generator(work_path, op_protos)
    call_pyboost_functions_py_generator(work_path, op_protos, doc_yaml_data)
    call_pyboost_functions_h_generator(work_path, op_protos)
    call_pyboost_functions_cpp_generator(work_path, op_protos)
    call_pyboost_overload_functions_cpp_generator(work_path, op_protos, mint_func_protos, alias_func_mapping)
    call_pyboost_internal_functions_h_generator(work_path, op_protos)
    call_pyboost_internal_functions_cpp_generator(work_path, op_protos)
    call_pyboost_internal_kernel_info_adapter_generator(work_path, op_protos)
    call_pyboost_grad_functions_cpp_generator(work_path, op_protos)
    call_pyboost_native_grad_functions_generator(work_path, op_protos)
    call_pyboost_op_cpp_code_generator(work_path, op_protos)
    # op splice
    call_pyboost_auto_grad_cpp_code_generator(work_path, op_protos)


def call_pyboost_auto_grad_cpp_code_generator(work_path, op_protos):
    call_auto_grad_impl_cc_generator(work_path, op_protos)
    call_auto_grad_reg_header_generator(work_path, op_protos)
    call_functions_header_generator(work_path, op_protos)
    call_functions_cc_generator(work_path, op_protos)


def call_auto_grad_impl_cc_generator(work_path, op_protos):
    generator = AutoGradImplGenerator()
    generator.generate(work_path, op_protos)


def call_auto_grad_reg_header_generator(work_path, op_protos):
    generator = AutoGradRegHeaderGenerator()
    generator.generate(work_path, op_protos)


def call_functions_header_generator(work_path, op_protos):
    generator = FunctionsHeaderGenerator()
    generator.generate(work_path, op_protos)


def call_functions_cc_generator(work_path, op_protos):
    generator = FunctionsGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_inner_prim_generator(work_path, op_protos):
    generator = PyboostInnerPrimGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_functions_py_generator(work_path, op_protos, doc_yaml_data):
    generator = PyboostFunctionsPyGenerator()
    generator.generate(work_path, op_protos, doc_yaml_data)


def call_pyboost_functions_h_generator(work_path, op_protos):
    generator = PyboostFunctionsHeaderGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_internal_functions_h_generator(work_path, op_protos):
    "gen internal op functions headers"
    generator = PyboostInternalFunctionsHeaderGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_internal_functions_cpp_generator(work_path, op_protos):
    "gen internal op functions sources"
    generator = PyboostInternalFunctionsCppGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_internal_kernel_info_adapter_generator(work_path, op_protos):
    "gen kernel info adapter for internal op"
    generator = PyboostKernelInfoAdapterGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_functions_cpp_generator(work_path, op_protos):
    impl_generator = PyboostFunctionsImplGenerator()
    impl_generator.generate(work_path, op_protos)
    generator = PyboostFunctionsGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_overload_functions_cpp_generator(work_path, op_protos, mint_func_protos, alias_func_mapping):
    generator = PyboostOverloadFunctionsGenerator()
    generator.generate(work_path, op_protos,
                       mint_func_protos, alias_func_mapping)


def call_pyboost_grad_functions_cpp_generator(work_path, op_protos):
    generator = PyboostGradFunctionsGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_native_grad_functions_generator(work_path, op_protos):
    h_generator = PyboostGradFunctionsHeaderGenerator()
    h_generator.generate(work_path, op_protos)

    cc_generator = PyboostGradFunctionsCppGenerator()
    cc_generator.generate(work_path, op_protos)


def call_pyboost_op_cpp_code_generator(work_path, op_protos):
    delete_residual_files(work_path, op_protos)
    call_PyboostCommonOpCppCodeGenerator(work_path, op_protos)
    call_PyboostOpHeaderGenerator(work_path, op_protos)
    call_merge_pyboost_op_cpp_code_generator(work_path, op_protos)
    call_PyboostOpRegisterCppCodeGenerator(work_path, op_protos)


def call_merge_pyboost_op_cpp_code_generator(work_path, op_protos):
    generator = PyboostOpFunctionGenerator()
    generator.generate(work_path, op_protos)


def call_PyboostCommonOpCppCodeGenerator(work_path, op_protos):
    generator = PyboostCommonOpHeaderGenerator()
    generator.generate(work_path, op_protos)


def call_PyboostOpHeaderGenerator(work_path, op_protos):
    """ generate pyboost op headers """
    generator = PyboostOpHeaderGenerator('ascend')
    generator.generate(work_path, op_protos)
    generator = PyboostInternalOpHeaderGenerator('ascend')
    generator.generate(work_path, op_protos)

    generator = PyboostOpHeaderGenerator('gpu')
    generator.generate(work_path, op_protos)

    generator = PyboostOpHeaderGenerator('cpu')
    generator.generate(work_path, op_protos)

    generator = PyboostOpHeaderGenerator('meta')
    generator.generate(work_path, op_protos)


def call_PyboostOpRegisterCppCodeGenerator(work_path, op_protos):
    op_register_cpp_generator = PyboostOpRegisterCppCodeGenerator()
    op_register_cpp_generator.generate(work_path, op_protos)
