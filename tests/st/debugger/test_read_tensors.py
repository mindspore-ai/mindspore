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
Read tensor test script for offline debugger APIs.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import mindspore.offline_debug.dbg_services as d
from tests.security_utils import security_off_wrap
from dump_test_utils import build_dump_structure, write_tensor_to_json

GENERATE_GOLDEN = False
tensor_json = []


def run_read_tensors(is_sync):
    if is_sync:
        test_name = "sync_read_tensors"
    else:
        test_name = "async_read_tensors"

    # input tensor with zero slot
    tensor1 = np.array([32.0, 4096.0], np.float32)
    name1 = "CudnnUniformReal.CudnnUniformReal-op391.0.0."
    info1 = d.TensorInfo(node_name="Default/CudnnUniformReal-op391",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=False)
    # input tensor with non-zero slot
    tensor2 = np.array([[0.0, 32.0, 4096.0], [4.5, 6.78, -11.0]], np.float32)
    name2 = "ReluGradV2.ReluGradV2-op406.0.0."
    info2 = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/"
                         "gradReLU/ReluGradV2-op406",
                         slot=1, iteration=1, rank_id=0, root_graph_id=0, is_output=False)
    # output tensor with zero slot
    tensor3 = np.array([[[7.963e-05, 4.750e-05, 2.587e-05],
                         [8.339e-05, 5.025e-05, 2.694e-05],
                         [8.565e-05, 5.156e-05, 2.658e-05]],
                        [[8.017e-05, 4.804e-05, 2.724e-05],
                         [8.392e-05, 5.126e-05, 2.843e-05],
                         [8.613e-05, 5.257e-05, 2.819e-05]],
                        [[7.617e-05, 3.827e-05, 5.305e-06],
                         [7.474e-05, 3.719e-05, 3.040e-06],
                         [7.081e-05, 3.338e-05, -2.086e-06]]], np.float32)
    name3 = "Conv2DBackpropFilter.Conv2DBackpropFilter-op424.0.0."
    info3 = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/conv5-Conv2d/"
                         "gradConv2D/Conv2DBackpropFilter-op424",
                         slot=0, iteration=1, rank_id=0, root_graph_id=0, is_output=True)
    # output tensor with non-zero slot
    tensor4 = np.array([2705090541, 1099111076, 4276637100, 3586562544, 890060077, 1869062900], np.float32)
    name4 = "ReLUV2.ReLUV2-op381.0.0."
    info4 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/ReLUV2-op381",
                         slot=1, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    tensor_name = [name1, name2, name3, name4]
    tensor_list = [tensor1, tensor2, tensor3, tensor4]
    tensor_info = [info1, info2, info3, info4]

    pwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir=pwd) as tmp_dir:
        temp_dir = build_dump_structure(tmp_dir, tensor_name, tensor_list, "Test", tensor_info)

        debugger_backend = d.DbgServices(dump_file_path=temp_dir)
        debugger_backend.initialize(net_name="Test", is_sync_mode=is_sync)
        tensor_data = debugger_backend.read_tensors(tensor_info)

        if GENERATE_GOLDEN:
            print_read_tensors(tensor_info, tensor_data, 0, True, test_name)
        else:
            compare_expect_actual_result(tensor_info, tensor_data, 0, test_name)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_sync_read_tensors():
    run_read_tensors(True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_read_tensors():
    run_read_tensors(False)


def compare_expect_actual_result(tensor_info_list, tensor_data_list, test_index, test_name):
    """Compare actual result with golden file."""
    pwd = os.getcwd()
    golden_file = os.path.realpath(os.path.join(pwd, "golden", test_name + "_expected.json"))
    with open(golden_file) as f:
        expected_list = json.load(f)
    for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
        tensor_id = "tensor_" + str(test_index + x + 1)
        expect_tensor = expected_list[x + test_index][tensor_id]
        actual_tensor = write_tensor_to_json(tensor_info, tensor_data)
        assert expect_tensor == actual_tensor


def print_read_tensors(tensor_info_list, tensor_data_list, test_index, is_print, test_name):
    """Print read tensors result if GENERATE_GOLDEN is True."""
    for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
        tensor_name = "tensor_" + str(test_index + x + 1)
        tensor = write_tensor_to_json(tensor_info, tensor_data)
        tensor_json.append({tensor_name: tensor})
    if is_print:
        with open(test_name + "_expected.json", "w") as dump_f:
            json.dump(tensor_json, dump_f, indent=4, separators=(',', ': '))
