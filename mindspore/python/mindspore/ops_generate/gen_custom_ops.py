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
Auto generate custom ops files.
"""

import logging
import argparse
from resources.resource_manager import ResourceManager
from resources.resource_list import ResourceType
from resources.yaml_loader import CustomOpDocYamlLoader
from op_def.ops_def_cc_generator import CustomOpsDefCcGenerator
from op_def_py.custom_op_prim_py_generator import CustomOpPrimPyGenerator
from op_def_py.op_def_py_generator import CustomOpDefPyGenerator
from common.op_proto import CustomOpProtoLoader


def get_config():
    """get config from user"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module_name", type=str, required=True)
    parser.add_argument("-i", "--input_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-d", "--doc_path", type=str, required=True)
    return parser.parse_args()


def generate_custom_op_def(module_name, input_path, doc_path, output_path):
    """Automatically generate all necessary files for custom operators."""
    resource_mgr = ResourceManager()
    resource_mgr.register_resource(CustomOpProtoLoader(input_path))
    op_protos = resource_mgr.get_resource(ResourceType.OP_PROTO)
    doc_dict = {}
    if doc_path != "":
        resource_mgr.register_resource(CustomOpDocYamlLoader(doc_path))
        doc_dict = resource_mgr.get_resource(ResourceType.OP_DOC_YAML)

    generator = CustomOpsDefCcGenerator()
    generator.generate(output_path, op_protos)
    generator = CustomOpPrimPyGenerator()
    generator.generate(output_path, module_name, op_protos, doc_dict, "gen")
    generator = CustomOpDefPyGenerator()
    generator.generate(output_path, op_protos, doc_dict, "gen")


def main():
    """main function"""
    args = get_config()
    generate_custom_op_def(args.module_name, args.input_path, args.doc_path, args.output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical("Auto generate failed, err info: %s", e)
        raise e
