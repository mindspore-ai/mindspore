# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""generate json desc for graph kernel ops"""
import json
import json.decoder as jd
import traceback
from mindspore import log as logger
import mindspore._extends.graph_kernel.expanders as expanders
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException


def create_expander(expand_info):
    """Create an expander according to op name"""
    def call_func(func, arg):
        return func(arg)
    op_name = str(expand_info['name'])
    if not hasattr(expanders, op_name):
        raise GraphKernelUnsupportedException("Generator do not support op: {}".format(op_name))
    expander = getattr(expanders, op_name)
    return call_func(expander, expand_info)


def extract_expand_info(kernel_info):
    """Convert the json into a more friendly format"""
    input_desc = []
    if 'input_desc' in kernel_info and kernel_info['input_desc']:
        for desc in kernel_info['input_desc']:
            input_desc += desc
    attrs = {}
    if 'attr' in kernel_info and kernel_info['attr']:
        for attr in kernel_info["attr"]:
            attrs[attr["name"]] = attr["value"]
    expand_info = {
        "name": kernel_info["name"],
        "input_desc": input_desc,
        "output_desc": kernel_info["output_desc"],
        "attr": attrs,
        "process": kernel_info["process"],
    }
    return expand_info


def get_op_expander(json_str: str):
    """get op expander by json info"""
    try:
        kernel_info = json.loads(json_str)
        expand_info = extract_expand_info(kernel_info)

        expander = create_expander(expand_info)
        graph = expander.run()

        # dump graph to json desc.
        desc = graph.dump()
        return json.dumps(desc)

    except jd.JSONDecodeError:
        logger.error("Failed to generate graph kernel op")
        logger.error(traceback.format_exc())
        return None
    except GraphKernelUnsupportedException as e:
        logger.info(e.message)
        return ""
