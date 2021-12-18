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
Read constant tensor test script for offline debugger APIs.
"""

import os
import json
import shutil
import numpy as np

import mindspore.offline_debug.dbg_services as d
from dump_test_utils import build_dump_structure_with_constant, write_tensor_to_json
from tests.security_utils import security_off_wrap


class TestOfflineReadConstantTensor:
    """Test reading constant tensor for offline debugger"""
    GENERATE_GOLDEN = False
    test_name = "read_constant_tensor"
    tensor_json = []
    temp_dir = ''

    @classmethod
    def setup_class(cls):
        """Init setup for offline read tensor test"""
        tensor1 = np.array([32.0, 4096.0], np.float32)
        name1 = "Parameter.data-1.0.0."
        info1 = d.TensorInfo(node_name="Default--data-1",
                             slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)
        tensor_name = [name1]
        cls.tensor_info = [info1]
        tensor_list = [tensor1]
        cls.temp_dir = build_dump_structure_with_constant(tensor_name, tensor_list, "Cst", cls.tensor_info)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir)

    @security_off_wrap
    def test_read_tensors(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Cst", is_sync_mode=True)
        tensor_data = debugger_backend.read_tensors(self.tensor_info)

        # Check the length of tensor data
        assert len(tensor_data) == 1
        if self.GENERATE_GOLDEN:
            self.print_read_tensors(self.tensor_info, tensor_data, 0, False)
        else:
            self.compare_expect_actual_result(self.tensor_info, tensor_data, 0)

    @security_off_wrap
    def test_read_invalid_constant_name_tensor(self):
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Cst", is_sync_mode=True)
        info = d.TensorInfo(node_name="Default/data-1",
                            slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)
        tensor_data = debugger_backend.read_tensors([info])

        assert len(tensor_data) == 1
        if self.GENERATE_GOLDEN:
            self.print_read_tensors([info], tensor_data, 1, True)
        else:
            self.compare_expect_actual_result([info], tensor_data, 1)

    def compare_expect_actual_result(self, tensor_info_list, tensor_data_list, test_index):
        """Compare actual result with golden file."""
        golden_file = os.path.realpath(os.path.join("../data/dump/gpu_dumps/golden/",
                                                    self.test_name + "_expected.json"))
        with open(golden_file) as f:
            expected_list = json.load(f)
        for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
            tensor_id = "tensor_" + str(test_index + x + 1)
            expect_tensor = expected_list[x + test_index][tensor_id]
            actual_tensor = write_tensor_to_json(tensor_info, tensor_data)
            assert expect_tensor == actual_tensor

    def print_read_tensors(self, tensor_info_list, tensor_data_list, test_index, is_print):
        """Print read tensors result if GENERATE_GOLDEN is True."""
        for x, (tensor_info, tensor_data) in enumerate(zip(tensor_info_list, tensor_data_list)):
            tensor_name = "tensor_" + str(test_index + x + 1)
            tensor = write_tensor_to_json(tensor_info, tensor_data)
            self.tensor_json.append({tensor_name: tensor})
        if is_print:
            with open(self.test_name + "_expected.json", "w") as dump_f:
                json.dump(self.tensor_json, dump_f, indent=4, separators=(',', ': '))
