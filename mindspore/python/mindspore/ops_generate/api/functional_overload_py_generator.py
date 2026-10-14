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
Module for generating C++ header files with operator name definitions.

This module defines the `OpsNameHGenerator` class, which produces C++ code to define
constants for operator names based on given prototypes.
"""

import os

import common.gen_constants as K
import common.gen_utils as gen_utils
import common.template as template
from common.template import Template

from common.base_generator import BaseGenerator


class FunctionalOverloadPyGenerator(BaseGenerator):
    """
    Class for generating C++ header files containing operator name constants.
    """

    def __init__(self):
        """
        Initializes the OpsNameHGenerator instance.
        """
        self.FUNCTIONAL_OVERLOAD_PY_TEMPLATE = template.FUNCTIONAL_OVERLOAD_PY_TEMPLATE

        self.mint_func_doc_yaml_dir_path = os.path.join(K.WORK_DIR, K.MS_MINT_FUNC_DOC_YAML_PATH)
        self.import_mint_template = Template("from mindspore._c_expression import _${cpp_func_name}_instance\n")
        self.mint_def_template = Template(
            'def ${mint_func_name}(*args, **kwargs):\n'
            '    r"""\n${docstr}\n    """\n'
            '    return _${cpp_func_name}_instance(*args, **kwargs)\n\n\n'
        )

    def generate(self, work_path, mint_func_protos_data, alias_api_mapping):
        """
        Generates python code for operator names and saves it to a header file.

        Args:
            mint_func_protos_data (dict): A dictionary mapping mint API names to their prototype data.
            function_doc_data (dict): A dictionary mapping function names to their docstring data.
            alias_api_mapping (dict): A dictionary mapping aliases to their prototype data.
        """
        function_doc_data = gen_utils.safe_load_yaml_from_dir(self.mint_func_doc_yaml_dir_path)
        validate_func_docs(mint_func_protos_data, function_doc_data, alias_api_mapping)
        import_mint_list, mint_init_list, mint_def_list, add_to_all_list = [], [], [], []
        for mint_api_name, _ in mint_func_protos_data.items():
            func_docstr = _format_docstring(function_doc_data.get(mint_api_name, {"description": ""})["description"])
            import_mint_list.append(self.import_mint_template.replace(cpp_func_name=mint_api_name))
            mint_def_list.append(self.mint_def_template.replace(mint_func_name=mint_api_name,
                                                                docstr=func_docstr,
                                                                cpp_func_name=mint_api_name))
            add_to_all_list.append(f'"{mint_api_name}",\n')
            if mint_api_name in alias_api_mapping:
                for alias_api_name in alias_api_mapping[mint_api_name]:
                    func_docstr = _format_docstring(
                        function_doc_data.get(alias_api_name, {"description": ""})["description"])
                    mint_def_list.append(self.mint_def_template.replace(mint_func_name=alias_api_name,
                                                                        docstr=func_docstr,
                                                                        cpp_func_name=mint_api_name))
                    add_to_all_list.append(f'"{alias_api_name}",\n')

        func_overload_py_file = self.FUNCTIONAL_OVERLOAD_PY_TEMPLATE.replace(import_mint_list=import_mint_list,
                                                                             mint_init_list=mint_init_list,
                                                                             mint_def_list=mint_def_list,
                                                                             add_to_all_list=add_to_all_list)
        save_path = os.path.join(work_path, K.MS_MINT_FUNC_OVERLOAD_PATH)
        file_name = "functional_overload.py"
        gen_utils.save_file(save_path, file_name, func_overload_py_file)


def _format_docstring(docstring, indent_size=4):
    if docstring is None:
        return None

    lines = docstring.split('\n')
    # Add 4 spaces to each line except first line
    formatted_lines = ([' ' * indent_size + lines[0]] +
                       [' ' * indent_size + line if line.strip() else line for line in lines[1:]])
    return '\n'.join(formatted_lines)


def validate_func_docs(mint_func_protos_data, function_doc_data, alias_api_mapping):
    """
    Ensure that the generated API includes corresponding docstrings; otherwise, raise an error to prompt the developer.
    """
    mint_api_names = set(mint_func_protos_data.keys())
    mint_doc_names = set(function_doc_data.keys())
    all_api_names = set()
    for mint_api_name in mint_api_names:
        if mint_api_name in alias_api_mapping:
            all_api_names = all_api_names.union(set(alias_api_mapping[mint_api_name]))
    all_api_names = all_api_names.union(mint_api_names)
    missing_docs = mint_doc_names - all_api_names
    if missing_docs:
        raise KeyError(f"Missing valid API references for the following doc names: {missing_docs}, "
                       f"please check if their doc.yaml files are defined in mindspore/ops/api_def/function_doc.")
