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
This module defines the OpHeaderFileGenerator class for generating header files for operator definitions.

The generator creates C++ header files that declare external operator definitions based on operator prototypes
and any additional operators provided. This is useful for managing operator interfaces in a consistent way.
"""

import os

from common import template
from common.template import Template
from common.gen_utils import save_file
import common.gen_constants as K
from common.base_generator import BaseGenerator


class OpsDefHGenerator(BaseGenerator):
    """
    Generates header files for operator definitions.

    This class is responsible for creating C++ header files that declare external operator definitions
    using templates. It processes a list of operator prototypes and can also include additional operators
    provided as extra arguments.
    """

    def __init__(self):
        """Initializes the OpHeaderFileGenerator and its templates."""
        super().__init__()
        self.extern_template = Template("OPS_API extern OpDef g${op_name};\n")
        self.GEN_OPS_DEF_HEADER_TEMPLATE = template.GEN_OPS_DEF_HEADER_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates the operator definition header file and saves it to the specified path.

        This method constructs the header content by creating extern declarations for each operator defined
        in the provided operator prototypes and any additional operators specified. The generated content
        is then saved to a C++ header file.

        Args:
            work_path (str): The directory path where the generated header file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None
        """
        extern_str = ''
        extra_ops = []
        for op_proto in op_protos:
            extern_str += self.extern_template.replace(op_name=op_proto.op_class.name)
            if op_proto.op_view:
                extra_ops.append(op_proto.op_class.name + "View")
        for class_name in extra_ops or []:
            extern_str += self.extern_template.replace(op_name=class_name)

        ops_header_file = self.GEN_OPS_DEF_HEADER_TEMPLATE.replace(extern_variable=extern_str)

        save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
        file_name = "gen_ops_def.h"
        save_file(save_path, file_name, ops_header_file)
