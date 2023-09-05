# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
        result = _load_repository(graph_desc, flags)
        if result:
            return json.dumps(result)
        target = graph_desc['process']
        comp = model.load_composite(graph_desc)
        subgraphs, graph_mode = model.split(comp.graph, target, flags)
        is_multi_graph = len(subgraphs) > 1
        graph_list = list(map(comp.dump, subgraphs))
        result = {"multi_graph": is_multi_graph,
                  "graph_desc": graph_list,
                  "graph_mode": graph_mode}
        if flags.get("dump_as_text", False):
            _dump_split_info(False, json_str, comp.graph, subgraphs, graph_mode, graph_list)
        return json.dumps(result)
    except jd.JSONDecodeError:
        logger.error(traceback.format_exc())
        return ""


def _load_repository(graph, flags):
    """Load repository if exists"""
    def check_repo(op, best_split, op_desc):
        if not isinstance(best_split, dict) or "group_num" not in best_split or "graph_mode" not in best_split \
                or "split_result" not in best_split:
            logger.warning("The graph split repository of {} should be a dict which contains 'group_num', 'graph_mode' "
                           "and 'split_result' field, but got {}".format(op, best_split))
            return False
        group_num = best_split["group_num"]
        split_result = best_split["split_result"]
        graph_mode = best_split["graph_mode"]
        if len(split_result) != len(op_desc):
            logger.warning("The graph split repository of {} is invalid, the size of 'split_result' should be equal to "
                           "the size of 'op_desc', but got len(split_result) = {} and len(op_desc) = {}".format(
                               op, len(split_result), len(op_desc)))
            return False
        if len(graph_mode) != group_num:
            logger.warning("The graph split repository of {} is invalid, the size of 'graph_mode' should be equal to "
                           "the 'group_num', but got len(graph_mode) = {} and group_num = {}".format(
                               op, len(graph_mode), group_num))
            return False
        if not all(0 <= i < group_num for i in split_result):
            logger.warning("The graph split repository of {} is invalid, all group id in 'split_result' should be "
                           "in range [0, {}), but got {}".format(op, group_num, split_result))
            return False
        if any(i not in ("basic", "composite") for i in graph_mode):
            logger.warning("The graph split repository of {} is invalid, the 'graph_mode' should be "
                           "'basic' or 'composite', but got {}".format(op, graph_mode))
            return False
        return True

    repository_path = flags.get("repository_path", "")
    if repository_path == "":
        return {}
    repo_file = os.path.join(os.path.realpath(repository_path), "repo_graph_split.json")
    try:
        with open(repo_file, "r") as f:
            repo_str = f.read()
    except IOError:
        logger.warning("Repository file {} is not accessible.".format(repo_file))
        return {}
    repo = json.loads(repo_str)
    op = graph.get("op", "")
    if op not in repo:
        logger.info("Op {} does not exist in repository {}, pattern model is used.".format(op, repo_file))
        return {}
    best_split = repo[op]
    op_desc = graph["op_desc"]
    if not check_repo(op, best_split, op_desc):
        return {}
    group_num = best_split["group_num"]
    groups = list([] for _ in range(group_num))
    subgraphs = list([] for _ in range(group_num))  # used for dump
    for op_idx, group_id in enumerate(best_split["split_result"]):
        groups[group_id].append(op_desc[op_idx])
        subgraphs[group_id].append(op_idx)
    groups = list({"op_desc": g} for g in groups)
    graph_mode = best_split["graph_mode"]
    result = {"multi_graph": len(groups) > 1,
              "graph_desc": groups,
              "graph_mode": graph_mode}
    if flags.get("dump_as_text", False):
        _dump_split_info(True, json.dumps(graph), model.load_composite(graph).graph, subgraphs, graph_mode, None)
    return result


def _dump_split_info(use_repo, graph_str, graph, subgraphs, graph_mode, graph_list):
    """Dump split info as text"""
    graph_kernel_dump_path = "graph_kernel_dump"
    utils.create_dir(graph_kernel_dump_path)
    filename = os.path.join(graph_kernel_dump_path, "graph_kernel_split_mode.%d.txt" % os.getpid())
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT, 0o600), "a+") as f:
        f.write("********** main graph: {} **********\n".format(graph.name))
        f.write("input json:\n{}\n".format(graph_str))
        f.write("graph desc:\n{}\n".format(str(graph)))
        if use_repo:
            f.write("(use repository result)\n")
        if len(subgraphs) > 1 or (not use_repo and subgraphs[0].stitch_info.has_stitch_op()):
            for i, g in enumerate(subgraphs):
                f.write("-------- subgraph {}, mode: {} --------\n".format(i, graph_mode[i]))
                if use_repo:
                    result = (str(graph.ops[op_idx]) for op_idx in subgraphs[i])
                    f.write("\n".join(result))
                    f.write("\n")
                else:
                    f.write("{}\n".format(str(g)))
                    f.write("json: {}\n".format(json.dumps(graph_list[i])))
        else:
            f.write("Graph unchanged.\n")
        f.write("\n")
