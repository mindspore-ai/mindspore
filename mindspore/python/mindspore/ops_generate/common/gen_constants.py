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
This module contains the constant strings used in generating ops files.

Constants:
    PY_LICENSE: License strings used for .py files
    CC_LICENSE: License strings used for .h/.cc files
    ......
    Other constant strings in the module are used for generation paths
"""

import os


WORK_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../../../../'))

PY_MS_DIR = "mindspore/python/mindspore"
PY_OPS_GEN_PATH = "mindspore/python/mindspore/ops_generate"
PY_AUTO_GEN_PATH = "mindspore/python/mindspore/ops/auto_generate"

# op_def
OP_DEF_AUTO_GENERATE_PATH = "op_def/auto_generate"
MS_OP_DEF_AUTO_GENERATE_PATH = "mindspore/ops/include/primitive/auto_generate"
MS_OP_DEF_AUTO_GENERATE_CC_PATH = "mindspore/ops/op_def/auto_generate"
MS_OP_DEF_YAML_PATH = "mindspore/ops/op_def/yaml"
MS_OP_DEF_FUNC_OP_YAML_PATH = "mindspore/ops/op_def/func_op"
MS_OP_DEPRECATED_DEF_YAML_PATH = "mindspore/ops/op_def/deprecated"
MS_OP_API_YAML_PATH = "mindspore/ops/api_def"
MS_TENSOR_METHOD_DOC_YAML_PATH = "mindspore/ops/api_def/method_doc"
MS_MINT_FUNC_DOC_YAML_PATH = "mindspore/ops/api_def/function_doc"
MS_MINT_FUNC_OVERLOAD_PATH = "mindspore/python/mindspore/ops"
PYBOOST_NATIVE_GRAD_FUNC_GEN_PATH = "mindspore/ccsrc/pynative/backward/op_grad/auto_generate"
PYBOOST_AUTO_GRAD_FUNC_GEN_PATH = "mindspore/ccsrc/pynative/forward/pyboost/auto_generate"
PIPELINE_PYBOOST_FUNC_GEN_PATH = "mindspore/ccsrc/pynative/forward/pyboost/auto_generate"
PIPELINE_PYBOOST_HEADER_FUNC_GEN_PATH = "mindspore/ccsrc/include/pynative/forward/pyboost/auto_generate"
FUNCTIONAL_OVERLOAD_GEN_PATH = "mindspore/ccsrc/frontend/operator/composite/auto_generate"
FUNCTIONAL_OVERLOAD_SIGNATURE_GEN_PATH = "mindspore/ccsrc/utils/operator/auto_generate"
PYBOOST_GRAD_FUNC_GEN_PATH = "mindspore/ccsrc/pynative/utils/pyboost/grad_functions/auto_generate"
TENSOR_FUNC_REGISTER_PATH = "mindspore/ccsrc/pynative/forward/pyboost/auto_generate"
TENSOR_API_PATH = "mindspore/ccsrc/pybind_api/pynative/tensor/tensor_api/auto_generate"
ADD_TENSOR_DOCS_PY_PATH = "mindspore/python/mindspore/common"
ADD_MINT_DOCS_PY_PATH = "mindspore/python/mindspore/mint"
TENSOR_PY_CC_PATH = "mindspore/ccsrc/pybind_api/pynative/tensor/tensor_register/auto_generate"

# yaml keys def
OP_KEYS = {'args', 'args_signature', 'returns', 'function', 'class', 'view', 'graph_view', 'dispatch', 'labels',
           'bprop_expander', 'non-differentiable', 'composite'}
ARG_KEYS = {'dtype', 'default', 'prim_init', 'type_cast', 'arg_handler', 'disable_tensor_to_scalar'}
RETURN_KEYS = {'dtype', 'inplace', 'type_cast'}
ARG_SIGNATURE_KEYS = {'rw_write', 'rw_read', 'rw_ref', 'dtype_group'}
CLASS_KEYS = {'name', 'disable'}
FUNCTION_KEYS = {'name', 'disable'}
DISPATCH_KEYS = {'enable', 'is_comm_op', 'Ascend', 'InternalOpAscend', 'GPU', 'CPU'}
TENSOR_FUNC_KEYS = {'op_yaml', 'py_method', 'kwonlyargs',
                    'varargs', 'disable_scalar_tensor',
                    'alias', 'Ascend', 'GPU', 'CPU', 'interface'}

# func signature parsing
ARG_HANDLER_MAP = {"to_2d_paddings": "int|tuple[int]|list[int]",
                   "dtype_to_type_id": "type",
                   "to_kernel_size": "int|tuple[int]|list[int]",
                   "to_strides": "int|tuple[int]|list[int]",
                   "str_to_enum": "str",
                   "to_pair": "int|tuple[int]|list[int]|float",
                   "to_dilations": "tuple[int]|list[int]|int",
                   "to_output_padding": "int|tuple[int]|list[int]",
                   "to_rates": "int|tuple[int]|list[int]"}
INPUT_ARGS_NAME = {"input", "x", "input_x"}
INPUT_NAME_MAP = {"DeprecatedExpandAs": "input"}

# infer
MS_OPS_FUNC_IMPL_PATH = "mindspore/ops/infer/ops_func_impl"
MS_OPS_COMM_FUNC_IMPL_PATH = "mindspore/ops/infer/ops_func_impl/communication"

# view
MS_OPS_VIEW_PATH = "mindspore/ops/include/view"

# kernel
MS_OPS_KERNEL_PATH = "mindspore/ops/kernel"
MS_PYBOOST_FUNCTIONS_HEADER_AUTO_GEN_PATH = "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate"
MS_PYBOOST_FUNCTIONS_AUTO_GEN_PATH = "mindspore/ccsrc/pynative/utils/pyboost/functions/auto_generate"
MS_COMMON_PYBOOST_KERNEL_PATH = os.path.join(MS_OPS_KERNEL_PATH, "common/pyboost")
MS_PYBOOST_BASE_PATH = "mindspore/ccsrc/pynative/utils/pyboost"
MS_PYBOOST_BASE_HEADER_PATH = "mindspore/ccsrc/include/pynative/utils/pyboost"
MS_PYBOOST_INTERNAL_FUNCTIONS_AUTO_GEN_PATH = os.path.join(MS_OPS_KERNEL_PATH,
                                                           "ascend/aclnn/pyboost_impl/internal/functions")
MS_INTERNAL_PYBOOST_GEN_PATH = "mindspore/ops/kernel/ascend/internal/pyboost/auto_gen"
MS_PLUGIN_INTERNAL_PATH = "mindspore/ops/kernel/ascend/internal"
MS_OPS_PYBOOST_INTERNAL = "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/internal"
