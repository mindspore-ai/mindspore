# Copyright 2024 Huawei Technologies Co., Ltd
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
import numpy as np
from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds
import mindspore as ms
from .util import Capture, capture

def dataset_generator():
    for i in range(1, 10):
        yield (
            np.ones((32, i), dtype=np.float32),
            np.zeros((32, i, i, 3), dtype=np.int32))


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x1, x2):
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        return x1, x2


class WrapNet(nn.Cell):
    def __init__(self, network, ds_helper):
        super(WrapNet, self).__init__()
        self.net = connect_network_with_dataset(network, ds_helper)

    def construct(self, *inputs):
        return self.net(*inputs)


def test_get_next_for_ge():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass GetNextForGE.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    cap = Capture('getnext_for_ge', 'GetNext')
    with capture(cap):

        network = Net()
        dataset = ds.GeneratorDataset(dataset_generator, ["data1", "data2"])

        t0 = Tensor(dtype=ms.float32, shape=[32, None])
        t1 = Tensor(dtype=ms.int32, shape=[32, None, None, 3])
        network.set_inputs(t0, t1)

        dataset_helper = DatasetHelper(dataset, dataset_sink_mode=True, sink_size=-1)
        wrap_net = WrapNet(network, dataset_helper)

        for inputs in dataset_helper:
            outputs = wrap_net(*inputs)
            print(outputs)

    patterns = ['Default/net-_DataWrapper/DynamicGetNextV2-op']
    cap.check_output(patterns)
