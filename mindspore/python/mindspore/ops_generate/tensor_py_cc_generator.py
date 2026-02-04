# Copyright 2025 Huawei Technologies Co., Ltd
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
Generates mindspore/ccsrc/pybind_api/pynative/tensor/tensor_py.cc which includes the CPython Tensor APIs.
"""

import os
import common.gen_constants as K
from common.gen_utils import save_file
from common import template
from common.template import Template
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils

class TensorPyCppGenerator(BaseGenerator):
    """
    This class is responsible for generating mindspore/ccsrc/pybind_api/pynative/tensor/tensor_register/
    auto_generate/tensor_py_gen.cc
    """
    def __init__(self):
        self.TENSOR_PY_CC_TEMPLATE = template.TENSOR_PY_CC_TEMPLATE
        self.TENSOR_PY_H_TEMPLATE = template.TENSOR_PY_H_TEMPLATE
        self.cpy_wrapper_template = Template("  DEFINE_TENSOR_METHOD_CPYWRAPPER(${pascal_api_name}) \\")
        self.tensor_api_def_template = Template(
            '{"${snake_api_name}"'
            ', (PyCFunction)TensorMethod${pascal_api_name}_CPyWrapper, METH_VARARGS | METH_KEYWORDS},'
        )

    def generate(self, work_path, tensor_method_protos, alias_func_mapping):
        """
        Generates the content for the helper file and saves it to the specified path.

        Args:
            work_path (str): The directory where the generated file will be saved.
            tensor_method_protos (dict): A dict mapping from Tensor func API names to their proto lists.
            alias_func_mapping (dict): A dictionary mapping function name to its alias function names.

        Returns:
            None
        """
        wrapper_defs = []
        tensor_api_defs = []
        for api_name, _ in tensor_method_protos.items():
            pascal_api_name = pyboost_utils.format_func_api_name(api_name)
            snake_api_name = api_name
            wrapper_defs.append(self.cpy_wrapper_template.replace(pascal_api_name=pascal_api_name))
            tensor_api_defs.append(
                self.tensor_api_def_template.replace(
                    snake_api_name=snake_api_name,
                    pascal_api_name=pascal_api_name
                )
            )
            if api_name in alias_func_mapping:
                alias_api_names = alias_func_mapping[api_name]
                for alias_api_name in alias_api_names:
                    snake_api_name = alias_api_name
                    tensor_api_defs.append(
                        self.tensor_api_def_template.replace(
                            snake_api_name=snake_api_name,
                            pascal_api_name=pascal_api_name
                        )
                    )

        # delete the ' \' for the last wrapper macro definition
        wrapper_defs[-1] = wrapper_defs[-1][:-2]

        file_str = self.TENSOR_PY_CC_TEMPLATE.replace(
            tensor_api_defs=tensor_api_defs
        )
        save_file(
            os.path.join(work_path, K.TENSOR_PY_CC_PATH),
            "tensor_py_gen.cc",
            file_str
        )

        file_str = self.TENSOR_PY_H_TEMPLATE.replace(CPyWrapper_defs=wrapper_defs)
        save_file(
            os.path.join(work_path, K.TENSOR_PY_CC_PATH),
            "tensor_py_gen.h",
            file_str
        )

def _format_api_name(api_name):
    has_suffix = api_name.endswith("_")
    parts = api_name.strip("_").split("_")
    formatted_api_name = "".join(part.capitalize() for part in parts)
    return formatted_api_name + '_' if has_suffix else formatted_api_name
