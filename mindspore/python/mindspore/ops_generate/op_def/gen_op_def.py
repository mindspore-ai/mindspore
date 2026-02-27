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
Generate operator definitions from ops.yaml
"""
import copy

from common import gen_constants as K
from resources.resource_list import ResourceType

from .ops_def_cc_generator import OpsDefCcGenerator
from .ops_def_h_generator import OpsDefHGenerator
from .ops_name_h_generator import OpsNameHGenerator
from .ops_primitive_h_generator import OpsPrimitiveHGenerator
from .lite_ops_cpp_generator import LiteOpsCcGenerator, LiteOpsHGenerator


def call_ops_def_cc_generator(work_path, op_protos):
    generator = OpsDefCcGenerator()
    generator.generate(work_path, op_protos)


def call_ops_def_h_generator(work_path, op_protos):
    generator = OpsDefHGenerator()
    generator.generate(work_path, op_protos)


def call_ops_primitive_h_generator(work_path, op_protos):
    generator = OpsPrimitiveHGenerator()
    generator.generate(work_path, op_protos)


def call_lite_ops_h_generator(work_path, op_protos):
    h_generator = LiteOpsHGenerator()
    h_generator.generate(work_path, op_protos)


def call_lite_ops_cc_generator(work_path, op_protos):
    generator = LiteOpsCcGenerator()
    generator.generate(work_path, op_protos)


def call_ops_name_h_generator(work_path, op_protos):
    h_generator = OpsNameHGenerator()
    h_generator.generate(work_path, op_protos)


def get_tensor_op_protos_with_deprecated(func_protos, op_protos):
    """
    Get op_protos with deprecated op_protos from func_protos.
    """
    tensor_op_protos = copy.deepcopy(op_protos)
    for _, item in func_protos.items():
        for func_proto in item:
            op_name = func_proto.op_proto.op_name
            if "deprecated" in func_proto.op_proto.op_name:
                func_proto.op_proto.op_class.name = ''.join(word.capitalize() for word in op_name.split('_'))
                if func_proto.op_proto.op_name[-1] == '_':
                    func_proto.op_proto.op_class.name += '_'
                tensor_op_protos.append(func_proto.op_proto)
    return tensor_op_protos


def generate_ops_def_files(resource_mgr):
    """
    Generate ops c++ file from yaml.
    """
    work_path = K.WORK_DIR
    op_protos = sorted(resource_mgr.get_resource(ResourceType.OP_PROTO), key=lambda op: op.op_name)
    tensor_method_protos = resource_mgr.get_resource(ResourceType.TENSOR_METHOD_PROTOS)
    # for generate tensor method deprecated in graph mode
    op_protos_with_deprecated = sorted(get_tensor_op_protos_with_deprecated(tensor_method_protos, op_protos),
                                       key=lambda op: op.op_name)
    call_ops_def_cc_generator(work_path, op_protos_with_deprecated)
    call_ops_def_h_generator(work_path, op_protos_with_deprecated)
    call_ops_primitive_h_generator(work_path, op_protos)
    call_lite_ops_h_generator(work_path, op_protos)
    call_lite_ops_cc_generator(work_path, op_protos)
    call_ops_name_h_generator(work_path, op_protos)
