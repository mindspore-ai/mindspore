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
Module for generating C++ header files for operator primitives.

This module defines the `OpsPrimitiveHGenerator` class, which creates C++ header files
containing definitions for operator primitives based on provided operator prototypes.
"""

import common.gen_constants as K
import common.gen_utils as gen_utils
import common.template as template
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils

OP_PRIM_OP_DEF_H = """
#ifndef MINDSPORE_OPS_PRIMITIVE_AUTO_GENERATE_GEN_OPS_PRIMITIVE_${suffix}_H_
#define MINDSPORE_OPS_PRIMITIVE_AUTO_GENERATE_GEN_OPS_PRIMITIVE_${suffix}_H_

#include "ir/primitive.h"
#include "mindapi/base/macros.h"
#include "primitive/auto_generate/gen_ops_name_${suffix}.h"

namespace mindspore::prim {
$ops_prim_gen
}  // namespace mindspore::prim
#endif  // MINDSPORE_OPS_PRIMITIVE_AUTO_GENERATE_GEN_OPS_PRIMITIVE_${suffix}_H_
"""

OP_PRIM_OP_DEF_CC = """

#include "primitive/auto_generate/gen_ops_primitive_${suffix}.h"

#include <memory>

namespace mindspore::prim {
$ops_prim_gen
}  // namespace mindspore::prim

"""


class OpsPrimitiveHGenerator(BaseGenerator):
    """
    This class generates the header file for operator primitives.
    """

    def __init__(self):
        """
        Initializes the generator with templates for operator primitive definitions.
        """
        self.op_prim_op_def_template = template.Template(OP_PRIM_OP_DEF_H)
        self.op_prim_op_def_cc_template = template.Template(OP_PRIM_OP_DEF_CC)
        self.op_def_h_template = template.Template(
            "OPS_API extern const PrimitivePtr kPrim${k_name_op};\n")
        self.op_def_template = template.Template(
            "const PrimitivePtr kPrim${k_name_op} = std::make_shared<Primitive>(ops::kName${k_name_op});\n")
        self.op_def_rw_template = template.Template(
            "const PrimitivePtr kPrim${k_name_op} = std::make_shared<Primitive>(ops::kName${k_name_op}, "
            "true, kPrimTypeBuiltIn, true);\n")

    def generate(self, work_path, op_protos):
        """
        Generates the header file content for operator primitives and saves it.

        Args:
            work_path (str): The directory to save the generated files.
            op_protos (list): A list of operator prototypes.

        Returns:
            None

        The method generates the content of the header file for each operator primitive
        defined in the 'op_protos' list and saves it to the specified work path.
        """
        import os
        import collections
        ops_prim_gen_dict = collections.defaultdict(list)
        ops_prim_cc_gen_dict = collections.defaultdict(list)

        for op_proto in op_protos:
            k_name_op = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
            first_char = k_name_op[0].lower()
            if op_proto.op_args_signature:
                if op_proto.op_args_signature.rw_write:
                    ops_prim_gen_dict[first_char].append(self.op_def_h_template.replace(k_name_op=k_name_op))
                    ops_prim_cc_gen_dict[first_char].append(self.op_def_rw_template.replace(k_name_op=k_name_op))
                    continue

            ops_prim_gen_dict[first_char].append(self.op_def_h_template.replace(k_name_op=k_name_op))
            ops_prim_cc_gen_dict[first_char].append(self.op_def_template.replace(k_name_op=k_name_op))

        for first_char, ops_prim_gen_list in ops_prim_gen_dict.items():
            op_prim_op_def = self.op_prim_op_def_template.replace(ops_prim_gen=ops_prim_gen_list, suffix=first_char)
            res_str = template.CC_LICENSE_STR + op_prim_op_def

            save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
            file_name = f"gen_ops_primitive_{first_char}.h"
            gen_utils.save_file(save_path, file_name, res_str)

        for first_char, ops_prim_gen_list in ops_prim_cc_gen_dict.items():
            op_prim_op_def = self.op_prim_op_def_cc_template.replace(ops_prim_gen=ops_prim_gen_list, suffix=first_char)
            res_str = template.CC_LICENSE_STR + op_prim_op_def

            save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_CC_PATH)
            file_name = f"gen_ops_primitive_{first_char}.cc"
            gen_utils.save_file(save_path, file_name, res_str)
