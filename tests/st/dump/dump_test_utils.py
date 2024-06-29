# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
import time
import glob
import csv
import numpy as np

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
            "Gradients/Default/network-WithLossCell/_backbone-ReLUReduceMeanDenseRelu/dense-Dense/gradBiasAdd/" \
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

e2e_dump_dict_3 = {
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
        "trans_flag": False,
        "slice_flag": 1,
        "slice_num": 20
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

async_dump_dict_acl = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 0,
        "kernels": [],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    }
}

async_dump_dict_acl_assign_ops_by_regex = {
    "common_dump_settings": {
        "dump_mode": 1,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 0,
        "kernels": ["name-regex(.+Add.*)"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    }
}

def generate_dump_json(dump_path, json_file_name, test_key, net_name='Net'):
    """
    Util function to generate dump configuration json file.
    """
    data = {}
    if test_key in ["test_async_dump", "test_async_dump_dataset_sink", "test_ge_dump"]:
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
    elif test_key in ("test_e2e_dump", "test_e2e_dump_trans_false"):
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_async_dump_net_multi_layer_mode1":
        data = async_dump_dict_2
        data["common_dump_settings"]["path"] = dump_path
    elif test_key in ("test_GPU_e2e_multi_root_graph_dump", "test_Ascend_e2e_multi_root_graph_dump"):
        data = e2e_dump_dict_2
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_Ascend_async_multi_root_graph_dump" or test_key == "test_ge_dump_net_multi_layer_mode1":
        data = async_dump_dict_3
        data["common_dump_settings"]["path"] = dump_path
    elif test_key in ["test_async_dump_npy", "test_ge_dump_npy", "test_acl_dump_complex"]:
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["file_format"] = "npy"
    elif test_key == "test_async_dump_bin":
        data = async_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["file_format"] = "bin"
    elif test_key in ["test_e2e_dump_trans_true", "test_e2e_dump_lenet", "test_e2e_dump_dynamic_shape"]:
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["trans_flag"] = True
    elif test_key == "test_e2e_dump_trans_true_op_debug_mode":
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["trans_flag"] = True
        data["common_dump_settings"]["op_debug_mode"] = 3
    elif test_key == "test_e2e_dump_save_kernel_args_true":
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["save_kernel_args"] = True
    elif test_key == "test_async_dump_net_multi_layer_mode1_npy":
        data = async_dump_dict_2
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["file_format"] = "npy"
    elif test_key == "test_e2e_dump_sample_debug_mode":
        data = e2e_dump_dict_3
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["trans_flag"] = True
    elif test_key == "test_acl_dump":
        data = async_dump_dict_acl
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_acl_dump_dynamic_shape":
        data = async_dump_dict_acl
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["file_format"] = "npy"
    elif test_key == "test_kbk_e2e_set_dump":
        data = e2e_dump_dict
        data["common_dump_settings"]["dump_mode"] = 2
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["trans_flag"] = True
    elif test_key == "test_kbk_e2e_dump_reg":
        data = e2e_dump_dict
        data["common_dump_settings"]["dump_mode"] = 1
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["kernels"] = ["name-regex(.+/Add[^/]*)"]
        data["e2e_dump_settings"]["trans_flag"] = True
    elif test_key == "test_exception_dump":
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["op_debug_mode"] = 4
        data["e2e_dump_settings"]["trans_flag"] = True
    elif test_key == "test_acl_dump_assign_ops_by_regex":
        data = async_dump_dict_acl_assign_ops_by_regex
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_acl_dump_exception":
        data = async_dump_dict_acl
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["op_debug_mode"] = 4
    else:
        raise ValueError(
            "Failed to generate dump json file. The test name value " + test_key + " is invalid.")
    data["common_dump_settings"]["net_name"] = net_name
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def generate_dump_json_with_overflow(dump_path, json_file_name, test_key, op):
    """
    Util function to generate dump configuration json file.
    """
    if test_key in ["test_async_dump", "test_ge_dump", "test_overflow_acl_dump"]:
        data = async_dump_dict
        common_dump_settings = data.get("common_dump_settings", "")
        if not isinstance(common_dump_settings, dict):
            raise ValueError("Common_dump_settings should be dict, but got %s." % type(common_dump_settings))
        common_dump_settings["path"] = dump_path
        common_dump_settings["op_debug_mode"] = op
    elif test_key == "test_async_dump_npy":
        data = async_dump_dict
        common_dump_settings = data.get("common_dump_settings", "")
        if not isinstance(common_dump_settings, dict):
            raise ValueError("Common_dump_settings should be dict, but got %s." % type(common_dump_settings))
        common_dump_settings["path"] = dump_path
        common_dump_settings["op_debug_mode"] = op
        common_dump_settings["file_format"] = "npy"
    else:
        raise ValueError(
            "Failed to generate dump json file. Overflow only support in async dump")
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def generate_statistic_dump_json(dump_path, json_file_name, test_key, saved_data, net_name='Net',
                                 statistic_category=None):
    """
    Util function to generate dump configuration json file for statistic dump.
    """
    data = {}
    if test_key in ["test_gpu_e2e_dump", "test_e2e_dump_dynamic_shape_custom_statistic"]:
        data = e2e_dump_dict
    elif test_key in ["test_async_dump", "test_ge_dump", "test_acl_dump"]:
        data = async_dump_dict
        data["common_dump_settings"]["input_output"] = 0
        data["common_dump_settings"]["file_format"] = "npy"
    elif test_key == "stat_calc_mode":
        data = e2e_dump_dict
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
    else:
        raise ValueError(
            "Failed to generate statistic dump json file. The test name value " + test_key + " is invalid.")
    data["common_dump_settings"]["path"] = dump_path
    data["common_dump_settings"]["saved_data"] = saved_data
    data["common_dump_settings"]["net_name"] = net_name
    if statistic_category:
        data["common_dump_settings"]["statistic_category"] = statistic_category
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


def check_dump_structure(dump_path, json_file_path, num_card, num_graph, num_iteration, root_graph_id=None,
                         test_iteration_id=None, execution_history=True):
    """
    Util to check if the dump structure is correct.
    """
    with open(json_file_path) as f:
        data = json.load(f)
    net_name = data["common_dump_settings"]["net_name"]
    assert os.path.isdir(dump_path)
    if root_graph_id is None:
        root_graph_id = [i for i in range(num_graph)]
    if test_iteration_id is None:
        test_iteration_id = [i for i in range(num_iteration)]
    for rank_id in range(num_card):
        rank_path = os.path.join(dump_path, "rank_" + str(rank_id))
        assert os.path.exists(rank_path)

        net_name_path = os.path.join(rank_path, net_name)
        assert os.path.exists(net_name_path)
        graph_path = os.path.join(rank_path, "graphs")
        assert os.path.exists(graph_path)
        execution_order_path = os.path.join(rank_path, "execution_order")
        assert os.path.exists(execution_order_path)

        for graph_id in range(num_graph):
            graph_pb_file = os.path.join(graph_path, "ms_output_trace_code_graph_" + str(graph_id) + ".pb")
            graph_ir_file = os.path.join(graph_path, "ms_output_trace_code_graph_" + str(graph_id) + ".ir")
            assert os.path.exists(graph_pb_file)
            assert os.path.exists(graph_ir_file)

            execution_order_file = os.path.join(execution_order_path, "ms_execution_order_graph_"
                                                + str(graph_id) + ".csv")
            assert os.path.exists(execution_order_file)
            if graph_id in root_graph_id:
                if execution_history:
                    execution_history_file = os.path.join(execution_order_path,
                                                          "ms_global_execution_order_graph_" + str(graph_id) + ".csv")
                    assert os.path.exists(execution_history_file)
                graph_id_path = os.path.join(net_name_path, str(graph_id))
                assert os.path.exists(graph_id_path)
                for iteration_id in test_iteration_id:
                    it_id_path = os.path.join(graph_id_path, str(iteration_id))
                    assert os.path.isdir(it_id_path)


def find_nth_pos(string, substring, n):
    start = string.find(substring)
    while n > 1 and start >= 0:
        start = string.find(substring, start + len(substring))
        n -= 1
    return start


def check_statistic_dump(dump_file_path):
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path) as f:
        reader = csv.DictReader(f)
        stats = list(reader)

        def get_add_node(statistic):
            return statistic['Op Type'] == 'Add'

        add_statistics = list(filter(get_add_node, stats))
        num_tensors = len(add_statistics)
        assert num_tensors == 3
        for tensor in add_statistics:
            if tensor['IO'] == 'input' and tensor['Slot'] == '0':
                assert tensor['Min Value'] == '1'
                assert tensor['Max Value'] == '6'
                assert tensor['L2Norm Value'] == '9.53939'
            elif tensor['IO'] == 'input' and tensor['Slot'] == '1':
                assert tensor['Min Value'] == '7'
                assert tensor['Max Value'] == '12'
                assert tensor['L2Norm Value'] == '23.6432'
            elif tensor['IO'] == 'output' and tensor['Slot'] == '0':
                assert tensor['Min Value'] == '8'
                assert tensor['Max Value'] == '18'
                assert tensor['L2Norm Value'] == '32.9242'


def check_data_dump(dump_file_path):
    output_name = "Add.*Add-op*.output.0.*.npy"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    output = np.load(real_path)
    expect = np.array([[8, 10, 12], [14, 16, 18]], np.float32)
    assert np.array_equal(output, expect)


def check_saved_data(iteration_path, saved_data):
    if not saved_data:
        return
    if saved_data in ('statistic', 'full'):
        check_statistic_dump(iteration_path)
    if saved_data in ('tensor', 'full'):
        check_data_dump(iteration_path)
    if saved_data == 'statistic':
        # assert only file is statistic.csv, tensor data is not saved
        assert len(os.listdir(iteration_path)) == 1
    elif saved_data == 'tensor':
        # assert only tensor data is saved, not statistics
        stat_path = os.path.join(iteration_path, 'statistic.csv')
        assert not os.path.isfile(stat_path)


def check_overflow_file(iteration_path, overflow_num, need_check):
    if not need_check:
        return overflow_num
    overflow_files = glob.glob(os.path.join(iteration_path, "Opdebug.Node_OpDebug.*.*.*"))
    overflow_num += len(overflow_files)
    return overflow_num


def check_iteration(iteration_id, num_iteration):
    if iteration_id.isdigit():
        assert int(iteration_id) < num_iteration


def check_ge_dump_structure(dump_path, num_iteration, device_num=1, check_overflow=False, saved_data=None,
                            check_data=True):
    overflow_num = 0
    for _ in range(3):
        if not os.listdir(dump_path):
            time.sleep(2)
    sub_paths = os.listdir(dump_path)
    assert sub_paths
    device_path_num = 0
    for sub_path in sub_paths:
        # on GE, the whole dump directory of one training is saved within a time path, like '20230822120819'
        if not (sub_path.isdigit() and len(sub_path) == 14):
            continue
        time_path = os.path.join(dump_path, sub_path)
        assert os.path.isdir(time_path)
        device_paths = os.listdir(time_path)
        device_path_num += len(device_paths)
        for device_path in device_paths:
            assert device_path.isdigit()
            abs_device_path = os.path.join(time_path, device_path)
            assert os.path.isdir(abs_device_path)
            model_names = os.listdir(abs_device_path)
            for model_name in model_names:
                model_path = os.path.join(abs_device_path, model_name)
                assert os.path.isdir(model_path)
                model_ids = os.listdir(model_path)
                for model_id in model_ids:
                    model_id_path = os.path.join(model_path, model_id)
                    assert os.path.isdir(model_id_path)
                    iteration_ids = os.listdir(model_id_path)
                    for iteration_id in iteration_ids:
                        check_iteration(iteration_id, num_iteration)
                        iteration_path = os.path.join(model_id_path, iteration_id)
                        assert os.path.isdir(iteration_path)
                        if check_data:
                            check_saved_data(iteration_path, saved_data)
                        overflow_num = check_overflow_file(iteration_path, overflow_num, check_overflow)
    assert device_path_num == device_num
    if check_overflow:
        assert overflow_num
