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
from tests.security_utils import security_off_wrap

GENERATE_GOLDEN = False
test_name = "sync_read_tensors"


@security_off_wrap
def test_sync_trans_false_read_tensors():

    # input tensor with zero slot
    tensor1 = np.array([32.0, 4096.0], np.float32)
    name1 = "CudnnUniformReal.CudnnUniformReal-op391.0.0."
    info1 = d.TensorInfo(node_name="Default/CudnnUniformReal-op391",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=False)
    # input tensor with non-zero slot
    tensor2 = np.array([[0.0, 32.0, 4096.0], [4.5, 6.78, -11.0]], np.float32)
    name2 = "ReluGradV2.ReluGradV2-op406.0.0."
    info2 = d.TensorInfo(node_name="Gradients/Default/network-WithLossCell/_backbone-AlexNet/gradReLU/ReluGradV2-op406",
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
    # inf tensor
    inf_tensor = np.array([[1., -np.inf, np.inf, -np.inf, np.inf], [np.inf, 1., -np.inf, np.inf, np.inf]], np.float32)
    inf_name = "Reciprocal.Reciprocal-op3.0.0."
    inf_info = d.TensorInfo(node_name="Default/Reciprocal-op3",
                            slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    tensor_name = [name1, name2, name3, name4]
    tensor_list = [tensor1, tensor2, tensor3, tensor4]
    tensor_info = [info1, info2, info3, info4]
    temp_dir = build_dump_structure(tensor_name, tensor_list, "alexnet", tensor_info)
    inf_dir = build_dump_structure([inf_name], [inf_tensor], "Inf", [inf_info])

    debugger_backend1 = d.DbgServices(dump_file_path=temp_dir)
    _ = debugger_backend1.initialize(net_name="alexnet", is_sync_mode=True)
    tensor_data = debugger_backend1.read_tensors(tensor_info)

    debugger_backend2 = d.DbgServices(dump_file_path=inf_dir)
    _ = debugger_backend2.initialize(net_name="Inf", is_sync_mode=True)
    tensor_data_inf = debugger_backend2.read_tensors([inf_info])
    tensor_info.extend([inf_info])
    tensor_data.extend(tensor_data_inf)

    shutil.rmtree(temp_dir)
    shutil.rmtree(inf_dir)
    print_read_tensors(tensor_info, tensor_data)

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
        f_write.write("tensor_info_" + str(x+1) + " attributes:\n")
        f_write.write("node name = " + tensor_info[x].node_name + "\n")
        f_write.write("slot = " + str(tensor_info[x].slot) + "\n")
        f_write.write("iteration = " + str(tensor_info[x].iteration) + "\n")
        f_write.write("rank_id = " + str(tensor_info[x].rank_id) + "\n")
        f_write.write("root_graph_id = " +
                      str(tensor_info[x].root_graph_id) + "\n")
        f_write.write("is_output = " +
                      str(tensor_info[x].is_output) + "\n")
        f_write.write("\n")
        f_write.write("tensor_data_" + str(x+1) + " attributes:\n")
        f_write.write("data (printed in uint8) = " + str(np.frombuffer(
            tensor_data[x].data_ptr, np.uint8, tensor_data[x].data_size)) + "\n")
        py_byte_size = len(tensor_data[x].data_ptr)
        c_byte_size = tensor_data[x].data_size
        if c_byte_size != py_byte_size:
            f_write.write("The python byte size of " + str(py_byte_size) +
                          " does not match the C++ byte size of " + str(c_byte_size) + "\n")
        f_write.write("size in bytes = " +
                      str(tensor_data[x].data_size) + "\n")
        f_write.write("debugger dtype = " + str(tensor_data[x].dtype) + "\n")
        f_write.write("shape = " + str(tensor_data[x].shape) + "\n")
    f_write.close()


if __name__ == "__main__":
    test_sync_trans_false_read_tensors()
