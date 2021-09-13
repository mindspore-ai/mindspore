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
import pytest
from dump_test_utils import compare_actual_with_expected
from tests.security_utils import security_off_wrap

GENERATE_GOLDEN = False
test_name = "async_sink_mode_true_read_tensors"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="needs updating")
@security_off_wrap
def test_async_sink_mode_true_read_tensors():
    debugger_backend = d.DbgServices(
        dump_file_path="/home/workspace/mindspore_dataset/dumps/async_sink_true/")

    _ = debugger_backend.initialize(net_name="alexnet", is_sync_mode=False)

    # output tensor with zero slot
    info1 = d.TensorInfo(node_name="Default/network-TrainOneStepCell/network-WithLossCell/_backbone-AlexNet/"
                                   "conv3-Conv2d/Conv2D-op169",
                         slot=0, iteration=2, device_id=0, root_graph_id=1, is_parameter=False)
    # output tensor with non-zero slot
    info2 = d.TensorInfo(node_name="Default/network-TrainOneStepCell/network-WithLossCell/_backbone-AlexNet/"
                                   "ReLUV2-op348",
                         slot=1, iteration=2, device_id=0, root_graph_id=1, is_parameter=False)

    tensor_info = [info1, info2]

    tensor_data = debugger_backend.read_tensors(tensor_info)

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
        f_write.write("-----------------------------------------------------------\n")
        f_write.write("tensor_info_" + str(x + 1) + " attributes:\n")
        f_write.write("node name =  " + tensor_info[x].node_name + "\n")
        f_write.write("slot =  " + str(tensor_info[x].slot) + "\n")
        f_write.write("iteration =  " + str(tensor_info[x].iteration) + "\n")
        f_write.write("device_id =  " + str(tensor_info[x].device_id) + "\n")
        f_write.write("root_graph_id =  " + str(tensor_info[x].root_graph_id) + "\n")
        f_write.write("is_parameter =  " + str(tensor_info[x].is_parameter) + "\n")
        f_write.write("\n")
        f_write.write("tensor_data_" + str(x + 1) + " attributes:\n")
        f_write.write("data (printed in uint8) =  " + str(np.frombuffer(
            tensor_data[x].data_ptr, np.uint8, tensor_data[x].data_size)) + "\n")
        py_byte_size = len(tensor_data[x].data_ptr)
        c_byte_size = tensor_data[x].data_size
        if c_byte_size != py_byte_size:
            f_write.write("The python byte size of  " + str(py_byte_size) +
                          " does not match the C++ byte size of  " + str(c_byte_size) + "\n")
        f_write.write("size in bytes =  " + str(tensor_data[x].data_size) + "\n")
        f_write.write("debugger dtype =  " + str(tensor_data[x].dtype) + "\n")
        f_write.write("shape =  " + str(tensor_data[x].shape) + "\n")
    f_write.close()
