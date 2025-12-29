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
import glob
import csv
import numpy as np
from mindspore import log as logger
import shutil

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

sync_kbyk_dump_dict = {
    "common_dump_settings": {
        "op_debug_mode": 0,
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "all",
        "input_output": 0,
        "saved_data": "statistic",
        "kernels": ["name-regex(.+Add.*)"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "statistic_category": ["max", "min", "avg", "l2norm"]
    },
    "e2e_dump_settings": {
        "enable": True,
        "trans_flag": True
    }
}

async_kbyk_dump_dict = {
    "common_dump_settings": {
        "op_debug_mode": 0,
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "all",
        "input_output": 0,
        "saved_data": "statistic",
        "kernels": ["name-regex(.+Add.*)"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "statistic_category": ["max", "min", "avg", "l2norm"]
    },
    "e2e_dump_settings": {
        "enable": False,
        "trans_flag": False
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

e2e_async_dump_dict = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0,
    },
    "e2e_dump_settings": {
        "enable": False,
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
        "sample_mode": 1,
        "sample_num": 20
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_dump_json(dump_path, json_file_name, test_key, net_name='Net', overflow_number=0):
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
    elif test_key in [
            "test_e2e_dump_trans_true_op_debug_mode",
            "test_e2e_dump_set_overflow_number",
            "test_dump_dynamic_net_sync_overflow_dump"
        ]:
        data = e2e_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["trans_flag"] = True
        data["common_dump_settings"]["op_debug_mode"] = 3
        data["common_dump_settings"]["overflow_number"] = overflow_number
        data["common_dump_settings"]["saved_data"] = "tensor"
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
    elif test_key == "test_acl_set_dump":
        data = async_dump_dict_acl
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["dump_mode"] = 2
    elif test_key == "test_sync_kbyk_dump":
        data = sync_kbyk_dump_dict
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_async_kbyk_dump":
        data = async_kbyk_dump_dict
        data["common_dump_settings"]["path"] = dump_path
    elif test_key == "test_lenet_sink_false_kbk_dump_sync":
        data = sync_kbyk_dump_dict
        data["common_dump_settings"]["path"] = dump_path
        data["common_dump_settings"]["net_name"] = 'lenet'
        data["common_dump_settings"]["saved_data"] = "tensor"
        data["common_dump_settings"]["input_output"] = 1
        data["common_dump_settings"]["iteration"] = "all"
        data["common_dump_settings"]["kernels"] = []
    else:
        raise ValueError(
            "Failed to generate dump json file. The test name value " + test_key + " is invalid.")
    data["common_dump_settings"]["net_name"] = net_name
    with open(json_file_name, 'w', encoding="utf-8") as f:
        json.dump(data, f)

def generate_statistic_dump_json(dump_path, json_file_name, test_key, saved_data, net_name='Net',
                                 statistic_category=None):
    """
    Util function to generate dump configuration json file for statistic dump.
    """
    data = {}
    if test_key in ["test_gpu_e2e_dump", "test_e2e_dump_dynamic_shape_custom_statistic", "test_e2e_dump"]:
        data = e2e_dump_dict
    elif test_key in ["test_async_dump", "test_ge_dump", "test_acl_dump"]:
        data = async_dump_dict
        data["common_dump_settings"]["input_output"] = 0
        data["common_dump_settings"]["file_format"] = "npy"
    elif test_key == "stat_calc_mode":
        data = e2e_dump_dict
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
    elif test_key == "test_e2e_async_dump":
        data = e2e_async_dump_dict
    elif test_key == "test_sync_kbyk_dump":
        data = sync_kbyk_dump_dict
    elif test_key == "test_async_kbyk_dump":
        data = async_kbyk_dump_dict
    elif test_key == "test_sync_kbyk_dump_resnet":
        data = e2e_dump_dict
    else:
        raise ValueError(
            "Failed to generate statistic dump json file. The test name value " + test_key + " is invalid.")
    data["common_dump_settings"]["path"] = dump_path
    data["common_dump_settings"]["saved_data"] = saved_data
    data["common_dump_settings"]["net_name"] = net_name
    if statistic_category:
        data["common_dump_settings"]["statistic_category"] = statistic_category
    with open(json_file_name, 'w', encoding="utf-8") as f:
        json.dump(data, f)


def check_dump_structure(dump_path, json_file_path, num_card, num_graph, num_iteration, root_graph_id=None,
                         test_iteration_id=None, execution_history=True):
    """
    Util to check if the dump structure is correct.
    """
    with open(json_file_path, encoding="utf-8") as f:
        data = json.load(f)
    net_name = data["common_dump_settings"]["net_name"]
    assert os.path.isdir(dump_path)
    if root_graph_id is None:
        root_graph_id = list(range(num_graph))
    if test_iteration_id is None:
        test_iteration_id = list(range(num_iteration))
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

def check_statistic_dump(dump_file_path):
    """
    Args:
        dump_path (str): dump文件路径。
        dump_config (dict): dump配置信息。
    """
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path, encoding="utf-8") as f:
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


def migrate_resnet50(case_path):
    """
    Args:
        case_path (str): dump文件路径。
    """
    try:
        shutil.copytree(os.path.join(os.path.dirname(ROOT_DIR), "networks", "models", "resnet50", "src"),
                        os.path.join(case_path, "src"))
    except Exception as e:
        logger.error(f"An error has occurred, the error message is {e}")
        raise e
    src_file = os.path.join(case_path, "src", "resnet.py")
    old_code = "def _weight_variable(shape, factor=0.01):"
    new_code = "def _weight_variable(shape, factor=26000):"
    replace_code(src_file, old_code, new_code)


def replace_code(source_file, old_str, new_str):
    """replace_code,适用整行代码替换"""
    dst_file = f'{source_file}.bak'
    with open(source_file, 'r', encoding="utf-8") as old_file, open(dst_file, 'w', encoding="utf-8") as new_file:
        for line in old_file:
            if old_str in line:
                line = line.replace(old_str, new_str)
            new_file.write(line)
    os.remove(source_file)
    os.rename('%s.bak' % source_file, source_file)
