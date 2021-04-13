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
    expander_list = {
        "AssignAdd": expanders.AssignAdd,
        "BiasAdd": expanders.BiasAdd,
        "BiasAddGrad": expanders.BiasAddGrad,
        "ClipByNormNoDivSum": expanders.ClipByNormNoDivSum,
        "DropoutGrad": expanders.DropoutGrad,
        "FusedAdam": expanders.FusedAdam,
        "FusedAdamWeightDecay": expanders.FusedAdamWeightDecay,
        "GeLU": expanders.GeLU,
        "GeLUGrad": expanders.GeLUGrad,
        "GkDropout": expanders.GkDropout,
        "LayerNorm": expanders.LayerNorm,
        "LayerNormGrad": expanders.LayerNormGrad,
        "LogSoftmax": expanders.LogSoftmax,
        "LogSoftmaxGrad": expanders.LogSoftmaxGrad,
        "MaximumGrad": expanders.MaximumGrad,
        "MinimumGrad": expanders.MinimumGrad,
        "ReduceMean": expanders.ReduceMean,
        "Softmax": expanders.Softmax,
        "Sigmoid": expanders.Sigmoid,
        "SigmoidGrad": expanders.SigmoidGrad,
        "SigmoidCrossEntropyWithLogits": expanders.SigmoidCrossEntropyWithLogits,
        "SigmoidCrossEntropyWithLogitsGrad": expanders.SigmoidCrossEntropyWithLogitsGrad,
        "SoftmaxCrossEntropyWithLogits": expanders.SoftmaxCrossEntropyWithLogits,
        "SqrtGrad": expanders.SqrtGrad,
        "Square": expanders.Square,
        "TanhGrad": expanders.TanhGrad,
        "Tile": expanders.Tile,
        "LambApplyOptimizerAssign": expanders.LambApplyOptimizerAssign,
    }
    op_name = str(expand_info['name'])
    if op_name not in expander_list:
        raise GraphKernelUnsupportedException("Generator do not support op: {}".format(op_name))
    return expander_list[op_name](expand_info)


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
