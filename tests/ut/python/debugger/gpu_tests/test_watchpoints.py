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
Watchpoints test script for offline debugger APIs.
"""

import os
import json
import shutil
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import build_dump_structure
from tests.security_utils import security_off_wrap


class TestOfflineWatchpoints:
    """Test watchpoint for offline debugger."""
    GENERATE_GOLDEN = False
    test_name = "watchpoints"
    watchpoint_hits_json = []
    temp_dir = ''

    @classmethod
    def setup_class(cls):
        """Init setup for offline watchpoints test"""
        name1 = "Conv2D.Conv2D-op369.0.0.1"
        tensor1 = np.array([[[-1.2808e-03, 7.7629e-03, 1.9241e-02],
                             [-1.3931e-02, 8.9359e-04, -1.1520e-02],
                             [-6.3248e-03, 1.8749e-03, 1.0132e-02]],
                            [[-2.5520e-03, -6.0005e-03, -5.1918e-03],
                             [-2.7866e-03, 2.5487e-04, 8.4782e-04],
                             [-4.6310e-03, -8.9111e-03, -8.1778e-05]],
                            [[1.3914e-03, 6.0844e-04, 1.0643e-03],
                             [-2.0966e-02, -1.2865e-03, -1.8692e-03],
                             [-1.6647e-02, 1.0233e-03, -4.1313e-03]]], np.float32)
        info1 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/conv1-Conv2d/Conv2D-op369",
                             slot=1, iteration=2, rank_id=0, root_graph_id=0, is_output=False)

        name2 = "Parameter.fc2.bias.0.0.2"
        tensor2 = np.array([-5.0167350e-06, 1.2509107e-05, -4.3148934e-06, 8.1415592e-06,
                            2.1177532e-07, 2.9952851e-06], np.float32)
        info2 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                             "Parameter[6]_11/fc2.bias",
                             slot=0, iteration=2, rank_id=0, root_graph_id=0, is_output=True)

        tensor3 = np.array([2.9060817e-07, -5.1009415e-06, -2.8662325e-06, 2.6036503e-06,
                            -5.1546101e-07, 6.0798648e-06], np.float32)
        info3 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                             "Parameter[6]_11/fc2.bias",
                             slot=0, iteration=3, rank_id=0, root_graph_id=0, is_output=True)

        name3 = "CudnnUniformReal.CudnnUniformReal-op391.0.0.3"
        tensor4 = np.array([-32.0, -4096.0], np.float32)
        info4 = d.TensorInfo(node_name="Default/CudnnUniformReal-op391",
                             slot=0, iteration=2, rank_id=0, root_graph_id=0, is_output=False)

        tensor_info = [info1, info2, info3, info4]
        tensor_name = [name1, name2, name2, name3]
        tensor_list = [tensor1, tensor2, tensor3, tensor4]
        cls.temp_dir = build_dump_structure(tensor_name, tensor_list, "Test", tensor_info)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir)

    @security_off_wrap
    def test_sync_add_remove_watchpoints_hit(self):
        # NOTES: watch_condition=6 is MIN_LT
        # watchpoint set and hit (watch_condition=6), then remove it
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=True)
        param = d.Parameter(name="param", disabled=False, value=0.0)
        _ = debugger_backend.add_watchpoint(watchpoint_id=1, watch_condition=6,
                                            check_node_list={"Default/network-WithLossCell/_backbone-AlexNet"
                                                             "/conv1-Conv2d/Conv2D-op369":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param])
        # add second watchpoint to check the watchpoint hit in correct order
        param1 = d.Parameter(name="param", disabled=False, value=10.0)
        _ = debugger_backend.add_watchpoint(watchpoint_id=2, watch_condition=6,
                                            check_node_list={"Default/CudnnUniformReal-op391":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param1])

        watchpoint_hits_test = debugger_backend.check_watchpoints(iteration=2)
        assert len(watchpoint_hits_test) == 2
        if self.GENERATE_GOLDEN:
            self.print_watchpoint_hits(watchpoint_hits_test, 0, False)
        else:
            self.compare_expect_actual_result(watchpoint_hits_test, 0)

        _ = debugger_backend.remove_watchpoint(watchpoint_id=1)
        watchpoint_hits_test_1 = debugger_backend.check_watchpoints(iteration=2)
        assert len(watchpoint_hits_test_1) == 1

    @security_off_wrap
    def test_sync_add_remove_watchpoints_not_hit(self):
        # watchpoint set and not hit(watch_condition=6), then remove
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=True)
        param = d.Parameter(name="param", disabled=False, value=-1000.0)
        _ = debugger_backend.add_watchpoint(watchpoint_id=2, watch_condition=6,
                                            check_node_list={"Default/network-WithLossCell/_backbone-AlexNet"
                                                             "/conv1-Conv2d/Conv2D-op369":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param])

        watchpoint_hits_test = debugger_backend.check_watchpoints(iteration=2)
        assert not watchpoint_hits_test
        _ = debugger_backend.remove_watchpoint(watchpoint_id=2)

    @security_off_wrap
    def test_sync_weight_change_watchpoints_hit(self):
        # NOTES: watch_condition=18 is CHANGE_TOO_LARGE
        # weight change watchpoint set and hit(watch_condition=18)
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=True)
        param_abs_mean_update_ratio_gt = d.Parameter(
            name="abs_mean_update_ratio_gt", disabled=False, value=0.0)
        param_epsilon = d.Parameter(name="epsilon", disabled=True, value=0.0)
        _ = debugger_backend.add_watchpoint(watchpoint_id=3, watch_condition=18,
                                            check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                                                             "Parameter[6]_11/fc2.bias":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": True
                                                              }}, parameter_list=[param_abs_mean_update_ratio_gt,
                                                                                  param_epsilon])

        watchpoint_hits_test = debugger_backend.check_watchpoints(iteration=3)
        assert len(watchpoint_hits_test) == 1
        if self.GENERATE_GOLDEN:
            self.print_watchpoint_hits(watchpoint_hits_test, 2, True)
        else:
            self.compare_expect_actual_result(watchpoint_hits_test, 2)

    @security_off_wrap
    def test_async_add_remove_watchpoint_hit(self):
        # watchpoint set and hit(watch_condition=6) in async mode, then remove
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=False)
        param = d.Parameter(name="param", disabled=False, value=0.0)
        _ = debugger_backend.add_watchpoint(watchpoint_id=1, watch_condition=6,
                                            check_node_list={"Default/network-WithLossCell/_backbone-AlexNet"
                                                             "/conv1-Conv2d/Conv2D-op369":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param])

        watchpoint_hits_test = debugger_backend.check_watchpoints(iteration=2)
        assert len(watchpoint_hits_test) == 1
        if not self.GENERATE_GOLDEN:
            self.compare_expect_actual_result(watchpoint_hits_test, 0)

        _ = debugger_backend.remove_watchpoint(watchpoint_id=1)
        watchpoint_hits_test_1 = debugger_backend.check_watchpoints(iteration=2)
        assert not watchpoint_hits_test_1

    @security_off_wrap
    def test_async_add_remove_watchpoints_not_hit(self):
        # watchpoint set and not hit(watch_condition=6) in async mode, then remove
        debugger_backend = d.DbgServices(dump_file_path=self.temp_dir)
        _ = debugger_backend.initialize(net_name="Test", is_sync_mode=False)
        param = d.Parameter(name="param", disabled=False, value=-1000.0)
        _ = debugger_backend.add_watchpoint(watchpoint_id=2, watch_condition=6,
                                            check_node_list={"Default/network-WithLossCell/_backbone-AlexNet"
                                                             "/conv1-Conv2d/Conv2D-op369":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param])

        watchpoint_hits_test = debugger_backend.check_watchpoints(iteration=2)
        assert not watchpoint_hits_test
        _ = debugger_backend.remove_watchpoint(watchpoint_id=2)

    def compare_expect_actual_result(self, watchpoint_hits_list, test_index):
        """Compare actual result with golden file."""
        golden_file = os.path.realpath(os.path.join("../data/dump/gpu_dumps/golden/",
                                                    self.test_name + "_expected.json"))
        with open(golden_file) as f:
            expected_list = json.load(f)
            for x, watchpoint_hits in enumerate(watchpoint_hits_list):
                test_id = "watchpoint_hit" + str(test_index+x+1)
                info = expected_list[x+test_index][test_id]
                assert watchpoint_hits.name == info['name']
                assert watchpoint_hits.slot == info['slot']
                assert watchpoint_hits.condition == info['condition']
                assert watchpoint_hits.watchpoint_id == info['watchpoint_id']
                assert watchpoint_hits.error_code == info['error_code']
                assert watchpoint_hits.rank_id == info['rank_id']
                assert watchpoint_hits.root_graph_id == info['root_graph_id']
                for p, _ in enumerate(watchpoint_hits.parameters):
                    parameter = "parameter" + str(p)
                    assert watchpoint_hits.parameters[p].name == info['paremeter'][p][parameter]['name']
                    assert watchpoint_hits.parameters[p].disabled == info['paremeter'][p][parameter]['disabled']
                    assert watchpoint_hits.parameters[p].value == info['paremeter'][p][parameter]['value']
                    assert watchpoint_hits.parameters[p].hit == info['paremeter'][p][parameter]['hit']
                    assert watchpoint_hits.parameters[p].actual_value == info['paremeter'][p][parameter]['actual_value']

    def print_watchpoint_hits(self, watchpoint_hits_list, test_index, is_print):
        """Print watchpoint hits."""
        for x, watchpoint_hits in enumerate(watchpoint_hits_list):
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
            watchpoint_hit = "watchpoint_hit" + str(test_index+x+1)
            self.watchpoint_hits_json.append({
                watchpoint_hit: {
                    'name': watchpoint_hits.name,
                    'slot': watchpoint_hits.slot,
                    'condition': watchpoint_hits.condition,
                    'watchpoint_id': watchpoint_hits.watchpoint_id,
                    'paremeter': parameter_json,
                    'error_code': watchpoint_hits.error_code,
                    'rank_id': watchpoint_hits.rank_id,
                    'root_graph_id': watchpoint_hits.root_graph_id
                }
            })
        if is_print:
            with open(self.test_name + "_expected.json", "w") as dump_f:
                json.dump(self.watchpoint_hits_json, dump_f, indent=4, separators=(',', ': '))
