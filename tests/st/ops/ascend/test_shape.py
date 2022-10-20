# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.ops import operations as P
from mindspore import Model, Tensor
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")

def dataset_generator():
    for i in range(1, 10):
        yield(np.ones((32, 2*i), dtype=np.float32), np.ones((32, 2*i), dtype=np.float32))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.unique = P.Unique()
        self.shape = P.TensorShape()
        self.reshape = P.Reshape()
        self.add = P.Add()

    def construct(self, x, y):
        val = self.add(x, y)
        size = self.shape(val)
        res = self.reshape(val, size)
        return res

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_shape():
    """
    Feature: dynamic shape
    Description: dynamic shape input data set
    Expectation: success
    """
    network = Net()
    dataset = ds.GeneratorDataset(dataset_generator, ["data1", "data2"])
    t0 = Tensor(dtype=ms.float32, shape=[32, None])
    t1 = Tensor(dtype=ms.float32, shape=[32, None])
    network.set_inputs(t0, t1)
    model = Model(network)
    model.train(1, dataset, sink_size=1)
