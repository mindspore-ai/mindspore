# Copyright 2020 Huawei Technologies Co., Ltd
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


def get_op_expander(json_str: str):
    """get op expander by json info"""
    try:
        kernel_info = json.loads(json_str)
        expand_info = kernel_info['expand_info']

        if 'name' not in expand_info:
            logger.error("expand info have no op name")
            return None
        if 'process' not in expand_info:
            logger.error("expand info have no processor info")
            return None

        processor = expand_info['process']
        op_name = str(expand_info['name']).lower()
        expand_op_func_name = 'expand_' + op_name
        if not hasattr(expanders, expand_op_func_name):
            logger.error("Generator do not support op: {}".format(op_name))
            return None
        expand_op_func = getattr(expanders, expand_op_func_name)
        # generate graph desc.
        graph = expand_op_func(expand_info)
        if graph is None:
            logger.error("Failed to generate graph of: {}".format(op_name))
            return None

        graph.set_processor(processor)

        # dump graph to json desc.
        desc = graph.dump()
        return json.dumps(desc)

    except jd.JSONDecodeError:
        logger.error("Failed to generate graph kernel op")
        logger.error(traceback.format_exc())
        return None
