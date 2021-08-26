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

import shutil
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import compare_actual_with_expected, build_dump_structure

GENERATE_GOLDEN = False
test_name = "sync_read_tensors_nonexist_node"


def test_sync_trans_read_tensors_nonexist_node():

    tensor1 = np.array([32.0, 4096.0], np.float32)
    name1 = "CudnnUniformReal.CudnnUniformReal-op391.0.0."
    info1 = d.TensorInfo(node_name="Default/CudnnUniformReal-op391",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=False)
    tensor2 = np.array([[0.0, 32.0, 4096.0], [4.5, 6.78, -11.0]], np.float32)
    name2 = "ReluGradV2.ReluGradV2-op406.0.0."
    info2 = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/gradReLU/ReluGradV2-op406",
                         slot=1, iteration=1, rank_id=0, root_graph_id=0, is_output=False)
    # non-existing tensor with wrong op name
    info3 = d.TensorInfo(node_name="Default/CudnnUniformReal-op390",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=False)

    # non-existing tensor with wrong iteration number
    info4 = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/gradReLU/ReluGradV2-op406",
                         slot=1, iteration=0, rank_id=0, root_graph_id=0, is_output=False)

    tensor_name = [name1, name2]
    tensor_create_info = [info1, info2]
    tensor_list = [tensor1, tensor2]
    temp_dir = build_dump_structure(tensor_name, tensor_list, "alexnet", tensor_create_info)
    tensor_check_info = [info3, info4]

    debugger_backend = d.DbgServices(dump_file_path=temp_dir)

    _ = debugger_backend.initialize(
        net_name="alexnet", is_sync_mode=True)

    tensor_data = debugger_backend.read_tensors(tensor_check_info)

    # Check the length of tensor list
    assert len(tensor_check_info) == 2
    assert len(tensor_data) == 2

    print_read_tensors(tensor_check_info, tensor_data)
    shutil.rmtree(temp_dir)
    if not GENERATE_GOLDEN:
        assert compare_actual_with_expected(test_name)


def print_read_tensors(tensor_info, tensor_data):
    """Print read tensors."""
    if GENERATE_GOLDEN:
        f_write = open(test_name + ".expected", "w")
    else:
        f_write = open(test_name + ".actual", "w")

    for x, _ in enumerate(tensor_info):
        f_write.write(
            "-----------------------------------------------------------\n")
        f_write.write("tensor_info_" + str(x + 1) + " attributes:\n")
        f_write.write("node name =  " + tensor_info[x].node_name + "\n")
        f_write.write("slot =  " + str(tensor_info[x].slot) + "\n")
        f_write.write("iteration =  " + str(tensor_info[x].iteration) + "\n")
        f_write.write("rank_id =  " + str(tensor_info[x].rank_id) + "\n")
        f_write.write("root_graph_id =  " +
                      str(tensor_info[x].root_graph_id) + "\n")
        f_write.write("is_output =  " +
                      str(tensor_info[x].is_output) + "\n")
        f_write.write("\n")
        f_write.write("tensor_data_" + str(x + 1) + " attributes:\n")
        f_write.write("data (printed in uint8) =  " + str(np.frombuffer(
            tensor_data[x].data_ptr, np.uint8, tensor_data[x].data_size)) + "\n")
        py_byte_size = len(tensor_data[x].data_ptr)
        c_byte_size = tensor_data[x].data_size
        if c_byte_size != py_byte_size:
            f_write.write("The python byte size of  " + str(py_byte_size) +
                          "  does not match the C++ byte size of  " + str(c_byte_size) + "\n")
        f_write.write("size in bytes =  " +
                      str(tensor_data[x].data_size) + "\n")
        f_write.write("debugger dtype =  " + str(tensor_data[x].dtype) + "\n")
        f_write.write("shape =  " + str(tensor_data[x].shape) + "\n")
    f_write.close()


if __name__ == "__main__":
    test_sync_trans_read_tensors_nonexist_node()
