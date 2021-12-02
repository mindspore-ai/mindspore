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
"""
Utils for testing dump feature.
"""

import json
import os

async_dump_dict = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 2,
        "kernels": ["Default/TensorAdd-op3"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    }
}

e2e_dump_dict = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    },
    "e2e_dump_settings": {
        "enable": True,
        "trans_flag": False
    }
}

async_dump_dict_2 = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "/tmp/async_dump/test_async_dump_net_multi_layer_mode1",
        "net_name": "test",
        "iteration": "0",
        "input_output": 2,
        "kernels": [
            "default/TensorAdd-op10",
            "Gradients/Default/network-WithLossCell/_backbone-ReLUReduceMeanDenseRelu/dense-Dense/gradBiasAdd/"\
            "BiasAddGrad-op8",
            "Default/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/SoftmaxCrossEntropyWithLogits-op5",
            "Default/optimizer-Momentum/tuple_getitem-op29",
            "Default/optimizer-Momentum/ApplyMomentum-op12"
        ],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    }
}

e2e_dump_dict_2 = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "all",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    },
    "e2e_dump_settings": {
        "enable": True,
        "trans_flag": False
    }
}

async_dump_dict_3 = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "all",
        "input_output": 2,
        "kernels": ["Default/TensorAdd-op3"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    }
}

def generate_dump_json(dump_path, json_file_name, test_key):
    """
    Util function to generate dump configuration json file.
    """
    if test_key == "test_async_dump":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_e2e_dump":
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_async_dump_net_multi_layer_mode1":
        data = async_dump_dict_2
        data["common_dump_settings"]["path"] = dump_path
    elif test_key in ("test_GPU_e2e_multi_root_graph_dump", "test_Ascend_e2e_multi_root_graph_dump"):
        data = e2e_dump_dict_2
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_Ascend_async_multi_root_graph_dump":
        data = async_dump_dict_3
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_async_dump_npy":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["file_format"] = "npy"
    elif test_key == "test_async_dump_bin":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["file_format"] = "bin"
    else:
        raise ValueError(
            "Failed to generate dump json file. The test name value " + test_key + " is invalid.")
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def generate_dump_json_with_overflow(dump_path, json_file_name, test_key, op):
    """
    Util function to generate dump configuration json file.
    """
    if test_key == "test_async_dump":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["op_debug_mode"] = op
    elif test_key == "test_async_dump_npy":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["op_debug_mode"] = op
        data["common_dump_settings"]["file_format"] = "npy"
    else:
        raise ValueError(
            "Failed to generate dump json file. Overflow only support in async dump")
    with open(json_file_name, 'w') as f:
        json.dump(data, f)

def generate_statistic_dump_json(dump_path, json_file_name, test_key, saved_data):
    """
    Util function to generate dump configuration json file for statistic dump.
    """
    if test_key == "test_gpu_e2e_dump":
        data = e2e_dump_dict
    elif test_key == "test_async_dump":
        data = async_dump_dict
        data["common_dump_settings"]["input_output"] = 0
        data["common_dump_settings"]["file_format"] = "npy"
    else:
        raise ValueError(
            "Failed to generate statistic dump json file. The test name value " + test_key + " is invalid.")
    data["common_dump_settings"]["path"] = dump_path
    data["common_dump_settings"]["saved_data"] = saved_data
    with open(json_file_name, 'w') as f:
        json.dump(data, f)

def generate_cell_dump_json(dump_path, json_file_name, test_key, dump_mode):
    """
    Util function to generate dump configuration json file.
    """
    if test_key == "test_async_dump":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["dump_mode"] = dump_mode
    else:
        raise ValueError(
            "Failed to generate dump json file. Overflow only support in async dump")
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def check_dump_structure(dump_path, json_file_path, num_card, num_graph, num_iteration):
    """
    Util to check if the dump structure is correct.
    """
    with open(json_file_path) as f:
        data = json.load(f)
    net_name = data["common_dump_settings"]["net_name"]
    assert os.path.isdir(dump_path)
    for rank_id in range(num_card):
        rank_path = os.path.join(dump_path, "rank_"+str(rank_id))
        assert os.path.exists(rank_path)

        net_name_path = os.path.join(rank_path, net_name)
        assert os.path.exists(net_name_path)
        graph_path = os.path.join(rank_path, "graphs")
        assert os.path.exists(graph_path)
        execution_order_path = os.path.join(rank_path, "execution_order")
        assert os.path.exists(execution_order_path)

        for graph_id in range(num_graph):
            graph_id_path = os.path.join(net_name_path, str(graph_id))
            assert os.path.exists(graph_id_path)

            graph_pb_file = os.path.join(graph_path, "ms_output_trace_code_graph_" + str(graph_id) + ".pb")
            graph_ir_file = os.path.join(graph_path, "ms_output_trace_code_graph_" + str(graph_id) + ".ir")
            assert os.path.exists(graph_pb_file)
            assert os.path.exists(graph_ir_file)

            execution_order_file = os.path.join(execution_order_path, "ms_execution_order_graph_"
                                                + str(graph_id) + ".csv")
            assert os.path.exists(execution_order_file)

            for iteration_id in range(num_iteration):
                it_id_path = os.path.join(graph_id_path, str(iteration_id))
                assert os.path.isdir(it_id_path)


def find_nth_pos(string, substring, n):
    start = string.find(substring)
    while n > 1 and start >= 0:
        start = string.find(substring, start + len(substring))
        n -= 1
    return start
