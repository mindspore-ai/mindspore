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
"""pynative hook"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops.operations as P
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark

class HookBase:
    def __init__(self):
        super().__init__()
        self.grad_input_list = []
        self.grad_output_list = []
        self.inputs_list = []
        self.outputs_list = []
        self.inputs_pre_list = []
        self.bprop_debug = False

    def record_hook(self, cell_id, grad_input, grad_output):
        for grad in grad_input:
            self.grad_input_list.append(grad)

        for grad in grad_output:
            self.grad_output_list.append(grad)

    def ms_record_hook(self, cell_id, grad_input, grad_output):
        for grad in grad_input:
            self.grad_output_list.append(grad)

        for grad in grad_output:
            self.grad_input_list.append(grad)

    def record_construct_hook(self, grad_out):
        for grad in grad_out:
            self.grad_input_list.append(grad)


class LeNet5(nn.Cell, HookBase):
    '''
    LeNet with construct hook
    '''
    def __init__(self):
        super().__init__()
        HookBase.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, pad_mode='valid')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, pad_mode='valid')
        self.fc1 = nn.Dense(in_channels=16 * 5 * 5, out_channels=120)
        self.fc2 = nn.Dense(in_channels=120, out_channels=84)
        self.fc3 = nn.Dense(in_channels=84, out_channels=10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.flatten = nn.Flatten()
        self.hook = P.HookBackward(self.record_construct_hook)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.hook(x)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=10.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_pynative_hook_lenet_train_with_construct_hook_and_register_cell_hook(mode):
    """
    Feature: pynative hook.
    Description: register backward hook for net.
    Expectation: success.
    """
    ms.set_context(mode=mode)
    input_me = Tensor(np.random.random((32, 1, 32, 32)).astype(np.float32))
    label_np = np.zeros([32, 10]).astype(np.float32)
    for i in range(0, 32):
        label_np[i][i % 10] = 1
    label = Tensor(label_np.astype(np.float32))
    net = LeNet5()
    net.conv1.register_backward_hook(net.ms_record_hook)
    loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    opt = Momentum(learning_rate=0.1, momentum=0.9, params=net.trainable_params())
    net_with_criterion = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_criterion, opt)
    train_network.set_train()
    for i in range(2):
        train_network(input_me, label)
    net(input_me)
