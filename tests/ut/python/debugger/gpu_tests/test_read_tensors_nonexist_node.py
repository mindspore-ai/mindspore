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
import shutil
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import build_dump_structure
from tests.security_utils import security_off_wrap


class TestOfflineReadNonExistTensor:
    """Test reading non exist tensor for offline debugger"""
    GENERATE_GOLDEN = False
    test_name = "read_tensors_nonexist_node"
    tensor_json = []
    temp_dir = ''

    @classmethod
    def setup_class(cls):
        """Init setup for offline read tensor test"""
        tensor1 = np.array([32.0, 4096.0], np.float32)
        name1 = "CudnnUniformReal.CudnnUniformReal-op391.0.0."
        info1 = d.TensorInfo(node_name="Default/CudnnUniformReal-op391",
                             slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=False)
        tensor2 = np.array([[0.0, 32.0, 4096.0], [4.5, 6.78, -11.0]], np.float32)
        name2 = "ReluGradV2.ReluGradV2-op406.0.0."
        info2 = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet"
                             "/gradReLU/ReluGradV2-op406",
                             slot=1, iteration=1, rank_id=0, root_graph_id=0, is_output=False)

        tensor_name = [name1, name2]
        tensor_info = [info1, info2]
        tensor_list = [tensor1, tensor2]
        cls.temp_dir = build_dump_structure(tensor_name, tensor_list, "Test", tensor_info)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir)

    @security_off_wrap
    def test_read_tensors_wrong_op_name(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(
            net_name="Test", is_sync_mode=True)

        # non-existing tensor with wrong op name
        info_nonexist = d.TensorInfo(node_name="Default/CudnnUniformReal-op390",
                                     slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=False)
        tensor_data = debugger_backend.read_tensors([info_nonexist])

        # Check the length of tensor data
        assert len(tensor_data) == 1
        if self.GENERATE_GOLDEN:
            self.print_read_tensors([info_nonexist], tensor_data, 0, False)
        else:
            self.compare_expect_actual_result([info_nonexist], tensor_data, 0)

    @security_off_wrap
    def test_read_tensors_wrong_iteration(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(
            net_name="Test", is_sync_mode=True)

        # non-existing tensor with wrong iteration number
        info_nonexist = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/"
                                     "gradReLU/ReluGradV2-op406",
                                     slot=1, iteration=0, rank_id=0, root_graph_id=0, is_output=False)
        tensor_data = debugger_backend.read_tensors([info_nonexist])

        assert len(tensor_data) == 1
        if self.GENERATE_GOLDEN:
            self.print_read_tensors([info_nonexist], tensor_data, 1, True)
        else:
            self.compare_expect_actual_result([info_nonexist], tensor_data, 1)

    @security_off_wrap
    def test_read_tensors_wrong_is_output(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(
            net_name="Test", is_sync_mode=True)

        # non-existing tensor with wrong is_output
        info_nonexist = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/"
                                     "gradReLU/ReluGradV2-op406",
                                     slot=1, iteration=1, rank_id=0, root_graph_id=0, is_output=True)
        tensor_data = debugger_backend.read_tensors([info_nonexist])

        assert len(tensor_data) == 1
        if self.GENERATE_GOLDEN:
            self.print_read_tensors([info_nonexist], tensor_data, 2, True)
        else:
            self.compare_expect_actual_result([info_nonexist], tensor_data, 2)

    def compare_expect_actual_result(self, tensor_info_list, tensor_data_list, test_index):
        """Compare actual result with golden file."""
        golden_file = os.path.realpath(os.path.join("../data/dump/gpu_dumps/golden/",
                                                    self.test_name + "_expected.json"))
        with open(golden_file) as f:
            expected_list = json.load(f)
        for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
            tensor_id = "tensor_"+ str(test_index+x+1)
            info = expected_list[x+test_index][tensor_id]
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
