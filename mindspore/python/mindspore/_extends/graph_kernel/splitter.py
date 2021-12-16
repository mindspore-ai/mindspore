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
"""GraphKernel splitter"""

import os
import json
import json.decoder as jd
import traceback
from mindspore import log as logger
from . import model
from . import utils


def split_with_json(json_str, flags_str):
    """Call cost model to split GraphKernel"""
    try:
        graph_desc = json.loads(json_str)
        flags = json.loads(flags_str)
        target = graph_desc['process']
        comp = model.load_composite(graph_desc)
        graph_split, graph_mode = model.split(comp.graph, target, flags)
        is_multi_graph = len(graph_split) > 1
        graph_list = list(map(comp.dump, graph_split))
        _reset_graphmode_for_inplaceassign(graph_list, graph_mode)
        result = {"multi_graph": is_multi_graph,
                  "graph_desc": graph_list,
                  "graph_mode": graph_mode}
        _dump_split_info(flags, json_str, comp.graph, graph_split, graph_mode)
        return json.dumps(result)
    except jd.JSONDecodeError:
        logger.error(traceback.format_exc())
        return None


def _reset_graphmode_for_inplaceassign(graph_list, graph_mode):
    """Operator with InplaceAssign should always be composite op"""
    for i, g in enumerate(graph_list):
        if any([op['name'] == 'InplaceAssign' for op in g['op_desc']]):
            graph_mode[i] = 'composite'


def _dump_split_info(flags, graph_json, graph_desc, subgraphs, graph_mode):
    """Dump split info as text"""
    if not flags.get("dump_as_text", False):
        return
    utils.create_dir(utils.GRAPH_KERNEL_DUMP_PATH)
    filename = os.path.join(utils.GRAPH_KERNEL_DUMP_PATH, "graph_kernel_split_mode.txt")
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT), "a+") as f:
        f.write("********** main graph: {} **********\n".format(graph_desc.name))
        f.write("input json:\n{}\n".format(graph_json))
        f.write("graph desc:\n{}\n".format(str(graph_desc)))
        if len(subgraphs) > 1 or subgraphs[0].stitch_info.has_stitch_op():
            for i, g in enumerate(subgraphs):
                f.write("-------- subgraph {}, mode: {} --------\n".format(i, graph_mode[i]))
                f.write("{}\n".format(str(g)))
        else:
            f.write("Graph unchanged.\n")
        f.write("\n")
