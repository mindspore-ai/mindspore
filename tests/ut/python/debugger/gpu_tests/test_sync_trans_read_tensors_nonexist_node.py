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

import mindspore.offline_debug.dbg_services as d
import numpy as np
from dump_test_utils import compare_actual_with_expected, skip_test

GENERATE_GOLDEN = False
test_name = "sync_trans_true_read_tensors_nonexist_node"


def test_sync_trans_read_tensors_nonexist_node():

    if skip_test():
        return

    debugger_backend = d.DbgServices(
        dump_file_path="../data/dump/gpu_dumps/sync_trans_true/alexnet")

    _ = debugger_backend.initialize(
        net_name="Network Name goes here!", is_sync_mode=True)

    # non-existing tensor with wrong op name
    info1 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/conv3-Conv2d/Conv2D-op318",
                         slot=0, iteration=2, device_id=0, root_graph_id=0, is_parameter=False)

    tensor_info = [info1]

    tensor_data = debugger_backend.read_tensors(tensor_info)

    # Check the length of tensor list
    assert len(tensor_info) == 1
    assert len(tensor_data) == 1

    print_read_tensors(tensor_info, tensor_data)
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
        f_write.write("device_id =  " + str(tensor_info[x].device_id) + "\n")
        f_write.write("root_graph_id =  " +
                      str(tensor_info[x].root_graph_id) + "\n")
        f_write.write("is_parameter =  " +
                      str(tensor_info[x].is_parameter) + "\n")
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
