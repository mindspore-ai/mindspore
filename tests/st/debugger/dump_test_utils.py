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
# ==============================================================================
"""
Utils for testing offline debugger.
"""

import os
import tempfile
import bisect
import csv
import numpy as np


def write_watchpoint_to_json(watchpoint_hits):
    parameter_json = []
    for p, _ in enumerate(watchpoint_hits.parameters):
        parameter = "parameter" + str(p)
        parameter_json.append({
            parameter: {
                'name': watchpoint_hits.parameters[p].name,
                'disabled': watchpoint_hits.parameters[p].disabled,
                'value': watchpoint_hits.parameters[p].value,
                'hit': watchpoint_hits.parameters[p].hit,
                'actual_value': watchpoint_hits.parameters[p].actual_value
            }
        })
    wp = {
        'name': watchpoint_hits.name,
        'slot': watchpoint_hits.slot,
        'condition': watchpoint_hits.condition,
        'watchpoint_id': watchpoint_hits.watchpoint_id,
        'parameter': parameter_json,
        'error_code': watchpoint_hits.error_code,
        'rank_id': watchpoint_hits.rank_id,
        'root_graph_id': watchpoint_hits.root_graph_id
        }
    return wp

def write_tensor_to_json(tensor_info, tensor_data):
    data = np.frombuffer(
        tensor_data.data_ptr, np.uint8, tensor_data.data_size).tolist()
    py_byte_size = len(tensor_data.data_ptr)
    c_byte_size = tensor_data.data_size
    if c_byte_size != py_byte_size:
        print("The python byte size of " + str(py_byte_size) +
              " does not match the C++ byte size of " + str(c_byte_size) + "\n")
    tensor = {
        'tensor_info': {
            'node_name': tensor_info.node_name,
            'slot': tensor_info.slot,
            'iteration': tensor_info.iteration,
            'rank_id': tensor_info.rank_id,
            'root_graph_id': tensor_info.root_graph_id,
            'is_output': tensor_info.is_output
        },
        'tensor_data': {
            'data': data,
            'size_in_bytes': tensor_data.data_size,
            'debugger_dtype': tensor_data.dtype,
            'shape': tensor_data.shape
        }
    }
    return tensor

def build_dump_structure(path, tensor_name_list, tensor_list, net_name, tensor_info_list):
    """Build dump file structure from tensor_list."""
    ranks_run_history = {}
    temp_dir = tempfile.mkdtemp(prefix=net_name, dir=path)
    for tensor_name, tensor, tensor_info in zip(tensor_name_list, tensor_list, tensor_info_list):
        slot = str(tensor_info.slot)
        iteration = str(tensor_info.iteration)
        rank_id = str(tensor_info.rank_id)
        root_graph_id = str(tensor_info.root_graph_id)
        is_output = str(tensor_info.is_output)
        graphs_run_history = ranks_run_history.get(rank_id)
        if graphs_run_history is None:
            graphs_run_history = {}
            ranks_run_history[rank_id] = graphs_run_history
        if root_graph_id not in graphs_run_history:
            graphs_run_history[root_graph_id] = [iteration]
        if iteration not in graphs_run_history[root_graph_id]:
            bisect.insort(graphs_run_history[root_graph_id], iteration)

        path = os.path.join(temp_dir, "rank_" + rank_id, net_name, root_graph_id, iteration)
        os.makedirs(path, exist_ok=True)
        if is_output == "True":
            file_name = f'{tensor_name}.output.{slot}.DefaultFormat.npy'
        else:
            file_name = f'{tensor_name}.input.{slot}.DefaultFormat.npy'
        full_path = os.path.join(path, file_name)
        np.save(full_path, tensor)
    build_global_execution_order(temp_dir, ranks_run_history)
    return temp_dir


def build_global_execution_order(path, ranks_run_history):
    """Build global execution order."""
    for rank_id in ranks_run_history.keys():
        exec_order_path = path + "/rank_" + rank_id + "/" + "execution_order"
        os.makedirs(exec_order_path, exist_ok=True)
        for graph in ranks_run_history[rank_id].keys():
            full_path = os.path.join(exec_order_path, "ms_global_execution_order_graph_" + graph + ".csv")
            with open(full_path, 'w+', newline='') as csv_file:
                write = csv.writer(csv_file)
                write.writerows(ranks_run_history[rank_id][graph])
