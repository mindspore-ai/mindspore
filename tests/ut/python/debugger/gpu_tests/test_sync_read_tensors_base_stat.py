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

import shutil
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import compare_actual_with_expected, build_dump_structure

GENERATE_GOLDEN = False
test_name = "sync_read_tensors_base_stat"


def test_sync_read_tensors_base_stat():

    value_tensor = np.array([[7.5, 8.56, -9.78], [10.0, -11.0, 0.0]], np.float32)
    name1 = "Add.Add-op4.0.0."
    info1 = d.TensorInfo(node_name="Default/Add-op4",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    inf_tensor = np.array([[1., -np.inf, np.inf, -np.inf, np.inf], [np.inf, 1., -np.inf, np.inf, np.inf]], np.float32)
    name2 = "Reciprocal.Reciprocal-op3.0.0."
    info2 = d.TensorInfo(node_name="Default/Reciprocal-op3",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    nan_tensor = np.array([-2.1754317, 1.9901361, np.nan, np.nan, -1.8091936], np.float32)
    name3 = "ReduceMean.ReduceMean-op92.0.0."
    info3 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-MockModel/ReduceMean-op92",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    tensor_info_1 = [info1]
    tensor_info_2 = [info2]
    tensor_info_3 = [info3]
    tensor_info = [info1, info2, info3]
    value_path = build_dump_structure([name1], [value_tensor], "Add", tensor_info_1)
    inf_path = build_dump_structure([name2], [inf_tensor], "Inf", tensor_info_2)
    nan_path = build_dump_structure([name3], [nan_tensor], "Nan", tensor_info_3)

    debugger_backend = d.DbgServices(
        dump_file_path=value_path, verbose=True)

    _ = debugger_backend.initialize(
        net_name="Add", is_sync_mode=True)

    debugger_backend_2 = d.DbgServices(
        dump_file_path=inf_path, verbose=True)

    _ = debugger_backend_2.initialize(
        net_name="Inf", is_sync_mode=True)

    debugger_backend_3 = d.DbgServices(
        dump_file_path=nan_path, verbose=True)

    _ = debugger_backend_3.initialize(
        net_name="Nan", is_sync_mode=True)

    tensor_base_data_list = debugger_backend.read_tensor_base(tensor_info_1)
    tensor_base_data_list_2 = debugger_backend_2.read_tensor_base(tensor_info_2)
    tensor_base_data_list.extend(tensor_base_data_list_2)
    tensor_base_data_list_3 = debugger_backend_3.read_tensor_base(tensor_info_3)
    tensor_base_data_list.extend(tensor_base_data_list_3)

    tensor_stat_data_list = debugger_backend.read_tensor_stats(tensor_info_1)
    tensor_stat_data_list_2 = debugger_backend_2.read_tensor_stats(tensor_info_2)
    tensor_stat_data_list.extend(tensor_stat_data_list_2)
    tensor_stat_data_list_3 = debugger_backend_3.read_tensor_stats(tensor_info_3)
    tensor_stat_data_list.extend(tensor_stat_data_list_3)

    shutil.rmtree(value_path)
    shutil.rmtree(inf_path)
    shutil.rmtree(nan_path)
    print_read_tensors(tensor_info, tensor_base_data_list, tensor_stat_data_list)
    if not GENERATE_GOLDEN:
        assert compare_actual_with_expected(test_name)


def print_read_tensors(tensor_info, tensor_base_data_list, tensor_stat_data_list):
    """Print read tensors info."""
    if GENERATE_GOLDEN:
        f_write = open(test_name + ".expected", "w")
    else:
        f_write = open(test_name + ".actual", "w")

    for x, _ in enumerate(tensor_info):
        f_write.write(
            "-----------------------------------------------------------\n")
        f_write.write("tensor_info_" + str(x+1) + " attributes:\n")
        f_write.write("node name =  " + tensor_info[x].node_name + "\n")
        f_write.write("slot =  " + str(tensor_info[x].slot) + "\n")
        f_write.write("iteration =  " + str(tensor_info[x].iteration) + "\n")
        f_write.write("rank_id =  " + str(tensor_info[x].rank_id) + "\n")
        f_write.write("root_graph_id =  " +
                      str(tensor_info[x].root_graph_id) + "\n")
        f_write.write("is_output =  " +
                      str(tensor_info[x].is_output) + "\n")
        f_write.write("\n")
        f_write.write("tensor_base_info:\n")
        f_write.write("size in bytes =  " +
                      str(tensor_base_data_list[x].data_size) + "\n")
        f_write.write("debugger dtype =  " + str(tensor_base_data_list[x].dtype) + "\n")
        f_write.write("shape =  " + str(tensor_base_data_list[x].shape) + "\n")

        f_write.write("\n")
        f_write.write("tensor_stat_info:\n")

        f_write.write("size in bytes =  " +
                      str(tensor_stat_data_list[x].data_size) + "\n")
        f_write.write("debugger dtype =  " + str(tensor_stat_data_list[x].dtype) + "\n")
        f_write.write("shape =  " + str(tensor_stat_data_list[x].shape) + "\n")
        f_write.write("is_bool =  " + str(tensor_stat_data_list[x].is_bool) + "\n")
        f_write.write("max_value =  " + str(tensor_stat_data_list[x].max_value) + "\n")
        f_write.write("min_value =  " + str(tensor_stat_data_list[x].min_value) + "\n")
        f_write.write("avg_value =  " + str(tensor_stat_data_list[x].avg_value) + "\n")
        f_write.write("count =  " + str(tensor_stat_data_list[x].count) + "\n")
        f_write.write("neg_zero_count =  " + str(tensor_stat_data_list[x].neg_zero_count) + "\n")
        f_write.write("pos_zero_count =  " + str(tensor_stat_data_list[x].pos_zero_count) + "\n")
        f_write.write("nan_count =  " + str(tensor_stat_data_list[x].nan_count) + "\n")
        f_write.write("neg_inf_count =  " + str(tensor_stat_data_list[x].neg_inf_count) + "\n")
        f_write.write("pos_inf_count =  " + str(tensor_stat_data_list[x].pos_inf_count) + "\n")
        f_write.write("zero_count =  " + str(tensor_stat_data_list[x].zero_count) + "\n")
    f_write.close()
