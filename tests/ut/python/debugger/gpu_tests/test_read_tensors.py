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
import shutil
import json
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import build_dump_structure
from tests.security_utils import security_off_wrap


class TestOfflineReadTensor:
    """Test read tensor for offline debugger."""
    GENERATE_GOLDEN = False
    test_name = "read_tensors"
    tensor_json = []
    temp_dir = ''

    @classmethod
    def setup_class(cls):
        """Init setup for offline read tensor test"""
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
        cls.tensor_info = [info1, info2, info3, info4]
        cls.temp_dir = build_dump_structure(tensor_name, tensor_list, "Test", cls.tensor_info)

        # inf tensor
        inf_tensor = np.array([[1., -np.inf, np.inf, -np.inf, np.inf],
                               [np.inf, 1., -np.inf, np.inf, np.inf]], np.float32)
        inf_name = "Reciprocal.Reciprocal-op3.0.0."
        cls.inf_info = d.TensorInfo(node_name="Default/Reciprocal-op3",
                                    slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

        cls.inf_dir = build_dump_structure([inf_name], [inf_tensor], "Inf", [cls.inf_info])

    @classmethod
    def teardown_class(cls):
        """Run after test this class."""
        shutil.rmtree(cls.temp_dir)
        shutil.rmtree(cls.inf_dir)

    @security_off_wrap
    def test_sync_read_tensors(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=True)
        tensor_data = debugger_backend.read_tensors(self.tensor_info)
        if self.GENERATE_GOLDEN:
            self.print_read_tensors(self.tensor_info, tensor_data, 0, False)
        else:
            self.compare_expect_actual_result(self.tensor_info, tensor_data, 0)

    @security_off_wrap
    def test_sync_read_inf_tensors(self):
        debugger_backend = d.DbgServices(dump_file_path=self.inf_dir)
        _ = debugger_backend.initialize(net_name="Inf", is_sync_mode=True)
        tensor_data_inf = debugger_backend.read_tensors([self.inf_info])

        if self.GENERATE_GOLDEN:
            self.print_read_tensors([self.inf_info], tensor_data_inf, 4, False)
        else:
            self.compare_expect_actual_result([self.inf_info], tensor_data_inf, 4)

    @security_off_wrap
    def test_async_read_tensors(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=False)
        tensor_data = debugger_backend.read_tensors(self.tensor_info)
        if not self.GENERATE_GOLDEN:
            self.compare_expect_actual_result(self.tensor_info, tensor_data, 0)

    @security_off_wrap
    def test_async_read_inf_tensors(self):
        debugger_backend = d.DbgServices(dump_file_path=self.inf_dir)
        _ = debugger_backend.initialize(net_name="Inf", is_sync_mode=False)
        tensor_data_inf = debugger_backend.read_tensors([self.inf_info])

        if not self.GENERATE_GOLDEN:
            self.compare_expect_actual_result([self.inf_info], tensor_data_inf, 4)

    def compare_expect_actual_result(self, tensor_info_list, tensor_data_list, test_index):
        """Compare actual result with golden file."""
        golden_file = os.path.realpath(os.path.join("../data/dump/gpu_dumps/golden/",
                                                    self.test_name + "_expected.json"))
        with open(golden_file) as f:
            expected_list = json.load(f)
        for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
            test_id = "tensor_"+ str(test_index+x+1)
            info = expected_list[x+test_index][test_id]
            assert tensor_info.node_name == info['tensor_info']['node_name']
            assert tensor_info.slot == info['tensor_info']['slot']
            assert tensor_info.iteration == info['tensor_info']['iteration']
            assert tensor_info.rank_id == info['tensor_info']['rank_id']
            assert tensor_info.root_graph_id == info['tensor_info']['root_graph_id']
            assert tensor_info.is_output == info['tensor_info']['is_output']
            actual_data = np.frombuffer(
                tensor_data.data_ptr, np.uint8, tensor_data.data_size).tolist()
            assert actual_data == info['tensor_data']['data']
            assert tensor_data.data_size == info['tensor_data']['size_in_bytes']
            assert tensor_data.dtype == info['tensor_data']['debugger_dtype']
            assert tensor_data.shape == info['tensor_data']['shape']

    def print_read_tensors(self, tensor_info_list, tensor_data_list, test_index, is_print):
        """Print read tensors result if GENERATE_GOLDEN is True."""
        for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
            tensor = "tensor_" + str(test_index+x+1)
            data = np.frombuffer(
                tensor_data.data_ptr, np.uint8, tensor_data.data_size).tolist()
            py_byte_size = len(tensor_data.data_ptr)
            c_byte_size = tensor_data.data_size
            if c_byte_size != py_byte_size:
                print("The python byte size of " + str(py_byte_size) +
                      " does not match the C++ byte size of " + str(c_byte_size) + "\n")
            self.tensor_json.append({
                tensor: {
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
            })
        if is_print:
            with open(self.test_name + "_expected.json", "w") as dump_f:
                json.dump(self.tensor_json, dump_f, indent=4, separators=(',', ': '))
