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
# ============================================================================
""" test jit in ge backend """
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.train import Model
import mindspore.dataset as ds

class ParamNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = ms.Parameter(Tensor(2, ms.float32), name="myname")

    @ms.jit
    def func(self, same_param):
        return same_param * self.param

    def construct(self, x):
        return self.func(self.param) * x


def generate_fake_dataset(batch_size, num_samples):
    data = np.random.rand(num_samples, 32).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples, 32)).astype(np.float32)
    return ds.NumpySlicesDataset((data, labels), shuffle=False).batch(batch_size)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_jit_ge_with_same_param():
    """
    Feature: jit with ge backend
    Description: test jit in ge backend with same params
    Expectation: success
    """
    net = ParamNet()
    loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    opt_fn = nn.Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
    dataset = generate_fake_dataset(batch_size=32, num_samples=256)
    model = Model(network=net, loss_fn=loss, optimizer=opt_fn)
    model.train(2, dataset)
