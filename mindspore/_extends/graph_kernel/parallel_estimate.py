# Copyright 2021 Huawei Technologies Co., Ltd
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
"""estimate parallel case"""
import json
import json.decoder as jd
import traceback
from mindspore import log as logger
from . import model

def estimate_ops(json_str: str):
    """Call costmodel to estimate ops."""
    try:
        json_obj = json.loads(json_str)
        graph_descs = json_obj["graph_desc"]
        graphs = []
        for gd in graph_descs:
            graphs.append(model.load_composite(gd).graph)
        estimation = model.parallel_estimate(graphs)
        res = (estimation.block_assign, estimation.gain,
               estimation.fusion_type, estimation.type_info)
        return res
    except jd.JSONDecodeError:
        logger.error(traceback.format_exc())
        return None

def estimate_calulation_amount(json_str: str):
    """Call costmodel to estimate calculation amount of op."""
    try:
        graph_desc = json.loads(json_str)
        comp = model.load_composite(graph_desc)
        estimation = model.parallel_estimate([comp.graph])
        return estimation.bottleneck
    except jd.JSONDecodeError:
        logger.error(traceback.format_exc())
        return None
