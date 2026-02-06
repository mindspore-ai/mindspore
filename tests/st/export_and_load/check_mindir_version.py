# Copyright 2025 Huawei Technologies Co., Ltd
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

"""check mindir version script."""

import os
import numpy as np
from mindspore import nn
from mindspore import jit
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.train.serialization import export, load


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.reshape = P.Reshape()

    @jit(backend="ms_backend")
    def construct(self, x):
        x = self.reshape(x, (self.batch_size, -1))
        return x


def run_check_mindir_version_no_warning():
    network = Net()
    network.set_train()
    input0 = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    output = network(input0)

    input1 = Tensor(np.zeros([32, 1, 32, 32]).astype(np.float32))
    export(network, input1, file_name="test_net", file_format='MINDIR')
    mindir_name = "test_net.mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    output_after_load = loaded_net(input0)
    assert np.allclose(output.asnumpy(), output_after_load.asnumpy())

    if os.path.exists(mindir_name):
        os.remove(mindir_name)


def run_check_mindir_version_warning(file):
    graph = load(file_name=file)
    nn.GraphCell(graph)


if __name__ == "__main__":
    file_name_list = ["old_version_mindir_1.mindir", "test_net_1_1.mindir", "test_net_2_1.mindir"]
    for file_name in file_name_list:
        file_path = os.path.realpath(os.path.dirname(__file__)) + "/exported_mindir/" + file_name
        run_check_mindir_version_warning(file_path)
    run_check_mindir_version_no_warning()
