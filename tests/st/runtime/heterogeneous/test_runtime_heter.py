# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test runtime heterogeneous scene."""
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

class LeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.weight1 = Parameter(Tensor(np.ones([6, 1, 5, 5]).astype(np.float16)), name="weight")
        self.weight2 = Parameter(Tensor(np.ones([16, 6, 5, 5]).astype(np.float16)), name="weight")

        self.relu = P.ReLU()
        self.relu_cpu = P.ReLU()
        self.relu_cpu.set_device("CPU")
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init=self.weight1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init=self.weight2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu_cpu(output)
        return output


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@test_utils.run_test_with_On
def test_lenet():
    """
    Feature: Runtime special format in the heterogeneous scene.
    Description: Test special format in the heterogeneous scene.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float16) * 0.01)
    net = LeNet()
    net(data)


class Lenet5(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, pad_mode='valid', weight_init='normal')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, pad_mode='valid', weight_init='normal')
        self.fc1 = nn.Dense(in_channels=16 * 5 * 5, out_channels=120, weight_init='normal', bias_init='zeros')
        self.fc2 = nn.Dense(in_channels=120, out_channels=84, weight_init='normal', bias_init='zeros')
        self.fc3 = nn.Dense(in_channels=84, out_channels=10, weight_init='normal', bias_init='zeros')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3


class FakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False,
                 fakedata_mode=FakeDataInitMode.RandomInit):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = np.float32
        self.label_data_type = np.float32
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        if use_parallel:
            init(backend_name='hccl')
            self.rank_size = get_group_size()
            self.rank_id = get_rank()

        self.total_batch_size = self.rank_batch_size * self.rank_size

        assert (self.size % self.total_batch_size) == 0

        self.total_batch_data_size = (
            self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        _ = num_epochs
        return self

    def __getitem__(self, batch_index):
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(
                self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.OnesInit:
            img = np.ones(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZerosInit:
            img = np.zeros(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UniqueInit:
            total_size = 1
            for i in self.total_batch_data_size:
                total_size = total_size * i
            img = np.reshape(np.arange(total_size) * 0.001,
                             self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)

        np.random.set_state(rng_state)
        img = img[self.rank_id]
        img_ret = img.astype(self.image_data_type)

        total_size = self.rank_batch_size * self.num_classes
        target = np.reshape(np.arange(total_size)*0.001,
                            (self.rank_batch_size, self.num_classes))
        return Tensor(img_ret), Tensor(target, dtype=ms.float32)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0

    def __next__(self):
        if self.batch_index * self.total_batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration


def train_lenet():
    net = Lenet5()
    net.conv1.conv2d.add_prim_attr("primitive_target", "CPU")
    net.conv1.bias_add.add_prim_attr("primitive_target", "CPU")
    net.fc2.matmul.add_prim_attr("primitive_target", "CPU")
    net.fc2.bias_add.add_prim_attr("primitive_target", "CPU")
    net.set_train()
    loss = SoftmaxCrossEntropyWithLogits(reduction="mean")
    opt = Momentum(learning_rate=0.0001, momentum=0.009,
                   params=filter(lambda x: x.requires_grad, net.get_parameters()))
    model = Model(net, loss, opt)
    dataset = FakeData(size=32, batch_size=32, image_size=(1, 32, 32), num_classes=10)
    model.train(1, dataset, dataset_sink_mode=False)
    inputs = Tensor(np.random.uniform(0.0, 1.0, size=[1, 1, 32, 32]).astype(np.float32))
    return model.predict(inputs)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@test_utils.run_test_with_On
def test_heter_lenet():
    """
    Feature: Runtime special format in the heterogeneous scene.
    Description: Test special format in the heterogeneous scene.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    out_ascend = train_lenet()
    print(out_ascend)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
    essential_mark='essential', mem_peak=2048.00)
@test_utils.run_test_with_On
def test_heter_lenet_pynative():
    """
    Feature: PyNative special format in the heterogeneous scene.
    Description: Test special format in the heterogeneous scene.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.set_seed(42)
    np.random.seed(42)
    out_ascend = train_lenet()
    expect_data = np.array([[2.6390028e-06, -1.0571928e-05, 5.6523363e-06, 5.9253930e-06,
                             9.7876073e-06, 3.1337552e-06, 5.2174191e-06, 9.4886109e-06,
                             7.1082345e-06, -3.6553743e-06]])
    assert np.allclose(out_ascend.asnumpy(), expect_data)
