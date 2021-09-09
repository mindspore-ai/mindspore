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
    else:
        raise ValueError(
            "Failed to generate dump json file. The test name value " + test_key + " is invalid.")
    with open(json_file_name, 'w') as f:
        json.dump(data, f)
