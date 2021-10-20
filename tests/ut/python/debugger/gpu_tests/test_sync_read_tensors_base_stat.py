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
Read tensor base and statistics test script for offline debugger APIs.
"""

import os
import shutil
import json
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import build_dump_structure
from tests.security_utils import security_off_wrap


class TestOfflineReadTensorBaseStat:
    """Test read tensor base stat for offline debugger"""
    GENERATE_GOLDEN = False
    test_name = "read_tensors_base_stat"
    tensor_json = []
    test_path = ''

    @classmethod
    def setup_class(cls):
        """Init setup for offline read tensor test"""
        value_tensor = np.array([[7.5, 8.56, -9.78], [10.0, -11.0, 0.0]], np.float32)
        name1 = "Add.Add-op4.0.0."
        info1 = d.TensorInfo(node_name="Default/Add-op4",
                             slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

        inf_tensor = np.array([[1., -np.inf, np.inf, -np.inf, np.inf],
                               [np.inf, 1., -np.inf, np.inf, np.inf]], np.float32)
        name2 = "Reciprocal.Reciprocal-op3.0.0."
        info2 = d.TensorInfo(node_name="Default/Reciprocal-op3",
                             slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

        nan_tensor = np.array([-2.1754317, 1.9901361, np.nan, np.nan, -1.8091936], np.float32)
        name3 = "ReduceMean.ReduceMean-op92.0.0."
        info3 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-MockModel/ReduceMean-op92",
                             slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

        invalid_tensor = np.array([[1.1, -2.2], [3.3, -4.4]], np.float32)
        name4 = "Add.Add-op1.0.0."
        info4 = d.TensorInfo(node_name="invalid_name_for_test",
                             slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

        cls.tensor_info_1 = [info1]
        cls.tensor_info_2 = [info2]
        cls.tensor_info_3 = [info3]
        cls.tensor_info_4 = [info4]
        cls.tensor_info = [info1, info2, info3, info4]
        cls.test_path = build_dump_structure([name1, name2, name3, name4],
                                             [value_tensor, inf_tensor, nan_tensor, invalid_tensor],
                                             "Test", cls.tensor_info)
        cls.debugger_backend = d.DbgServices(dump_file_path=cls.test_path)
        _ = cls.debugger_backend.initialize(net_name="Test", is_sync_mode=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_path)

    @security_off_wrap
    def test_read_value_tensors_base_stat(self):
        tensor_base_data_list = self.debugger_backend.read_tensor_base(self.tensor_info_1)
        tensor_stat_data_list = self.debugger_backend.read_tensor_stats(self.tensor_info_1)

        if self.GENERATE_GOLDEN:
            self.print_read_tensors(self.tensor_info_1, tensor_base_data_list, tensor_stat_data_list, 0, False)
        else:
            self.compare_expect_actual_result(self.tensor_info_1, tensor_base_data_list, tensor_stat_data_list, 0)

    @security_off_wrap
    def test_read_inf_tensors_base_stat(self):
        tensor_base_data_list = self.debugger_backend.read_tensor_base(self.tensor_info_2)
        tensor_stat_data_list = self.debugger_backend.read_tensor_stats(self.tensor_info_2)

        if self.GENERATE_GOLDEN:
            self.print_read_tensors(self.tensor_info_2, tensor_base_data_list, tensor_stat_data_list, 1, False)
        else:
            self.compare_expect_actual_result(self.tensor_info_2, tensor_base_data_list, tensor_stat_data_list, 1)

    @security_off_wrap
    def test_read_nan_tensors_base_stat(self):
        tensor_base_data_list = self.debugger_backend.read_tensor_base(self.tensor_info_3)
        tensor_stat_data_list = self.debugger_backend.read_tensor_stats(self.tensor_info_3)

        if self.GENERATE_GOLDEN:
            self.print_read_tensors(self.tensor_info_3, tensor_base_data_list, tensor_stat_data_list, 2, False)
        else:
            self.compare_expect_actual_result(self.tensor_info_3, tensor_base_data_list, tensor_stat_data_list, 2)

    @security_off_wrap
    def test_read_inv_tensors_base_stat(self):
        tensor_base_data_list = self.debugger_backend.read_tensor_base(self.tensor_info_4)
        tensor_stat_data_list = self.debugger_backend.read_tensor_stats(self.tensor_info_4)

        if self.GENERATE_GOLDEN:
            self.print_read_tensors(self.tensor_info_4, tensor_base_data_list, tensor_stat_data_list, 3, True)
        else:
            self.compare_expect_actual_result(self.tensor_info_4, tensor_base_data_list, tensor_stat_data_list, 3)

    def compare_expect_actual_result(self, tensor_info, tensor_base_data_list, tensor_stat_data_list, test_index):
        """Compare actual result with golden file."""
        golden_file = os.path.realpath(os.path.join("../data/dump/gpu_dumps/golden/",
                                                    self.test_name + "_expected.json"))
        with open(golden_file) as f:
            expected_list = json.load(f)

        for x, (tensor_info_item, tensor_base, tensor_stat) in enumerate(zip(tensor_info,
                                                                             tensor_base_data_list,
                                                                             tensor_stat_data_list)):
            test_id = "test"+ str(test_index+x+1)
            info_json = expected_list[x+test_index][test_id]['tensor_info']
            base_json = expected_list[x+test_index][test_id]['tensor_base_info']
            stat_json = expected_list[x+test_index][test_id]['tensor_stat_info']
            assert tensor_info_item.node_name == info_json['node_name']
            assert tensor_info_item.slot == info_json['slot']
            assert tensor_info_item.iteration == info_json['iteration']
            assert tensor_info_item.rank_id == info_json['rank_id']
            assert tensor_info_item.root_graph_id == info_json['root_graph_id']
            assert tensor_info_item.is_output == info_json['is_output']
            assert tensor_base.data_size == base_json['size_in_bytes']
            assert tensor_base.dtype == base_json['debugger_dtype']
            assert tensor_base.shape == base_json['shape']
            assert tensor_stat.data_size == stat_json['size_in_bytes']
            assert tensor_stat.dtype == stat_json['debugger_dtype']
            assert tensor_stat.shape == stat_json['shape']
            assert tensor_stat.is_bool == stat_json['is_bool']
            assert tensor_stat.max_value == stat_json['max_vaue']
            assert tensor_stat.min_value == stat_json['min_value']
            assert tensor_stat.avg_value == stat_json['avg_value']
            assert tensor_stat.count == stat_json['count']
            assert tensor_stat.neg_zero_count == stat_json['neg_zero_count']
            assert tensor_stat.pos_zero_count == stat_json['pos_zero_count']
            assert tensor_stat.nan_count == stat_json['nan_count']
            assert tensor_stat.neg_inf_count == stat_json['neg_inf_count']
            assert tensor_stat.pos_inf_count == stat_json['pos_inf_count']
            assert tensor_stat.zero_count == stat_json['zero_count']

    def print_read_tensors(self, tensor_info, tensor_base_data_list, tensor_stat_data_list, test_index, is_print):
        """Print read tensors info."""
        for x, (tensor_info_item, tensor_base, tensor_stat) in enumerate(zip(tensor_info,
                                                                             tensor_base_data_list,
                                                                             tensor_stat_data_list)):
            test_name = "test" + str(test_index+x+1)
            self.tensor_json.append({
                test_name: {
                    'tensor_info': {
                        'node_name': tensor_info_item.node_name,
                        'slot': tensor_info_item.slot,
                        'iteration': tensor_info_item.iteration,
                        'rank_id': tensor_info_item.rank_id,
                        'root_graph_id': tensor_info_item.root_graph_id,
                        'is_output': tensor_info_item.is_output
                    },
                    'tensor_base_info': {
                        'size_in_bytes': tensor_base.data_size,
                        'debugger_dtype': tensor_base.dtype,
                        'shape': tensor_base.shape
                    },
                    'tensor_stat_info': {
                        'size_in_bytes': tensor_stat.data_size,
                        'debugger_dtype': tensor_stat.dtype,
                        'shape': tensor_stat.shape,
                        'is_bool': tensor_stat.is_bool,
                        'max_vaue': tensor_stat.max_value,
                        'min_value': tensor_stat.min_value,
                        'avg_value': tensor_stat.avg_value,
                        'count': tensor_stat.count,
                        'neg_zero_count': tensor_stat.neg_zero_count,
                        'pos_zero_count': tensor_stat.pos_zero_count,
                        'nan_count': tensor_stat.nan_count,
                        'neg_inf_count': tensor_stat.neg_inf_count,
                        'pos_inf_count': tensor_stat.pos_inf_count,
                        'zero_count': tensor_stat.zero_count
                    }
                }})
        if is_print:
            with open(self.test_name + "_expected.json", "w") as dump_f:
                json.dump(self.tensor_json, dump_f, indent=4, separators=(',', ': '))
