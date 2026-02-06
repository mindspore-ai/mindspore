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
# ============================================================================
"""
test compile cache in lenet with jit
"""
import numpy as np
from mindspore import context, nn, Tensor, jit
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P


class LeNet(nn.Cell):
    """LeNet
    Args:
        None

    Inputs:
        input_x (Tensor): Input tensor of shape (batch_size, 1, height, width).

    Returns:
        Tensor, output tensor of shape (batch_size, 10).

    Examples:
        >>> net = LeNet()
        >>> input_x = Tensor(np.random.randn(32, 1, 32, 32), mstype.float32)
        >>> output = net(input_x)

    Note:
        The batch_size is fixed to 32 in this implementation.
    """
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()
        self.batch_size = 32

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init="normal")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init="normal")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120, weight_init="normal", bias_init="zeros")
        self.fc2 = nn.Dense(120, 84, weight_init="normal", bias_init="zeros")
        self.fc3 = nn.Dense(84, 10, weight_init="normal", bias_init="zeros")

    @jit
    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    input_label = Tensor(np.ones([32]).astype(np.int32))
    lenet = LeNet()

    learning_rate = 0.01
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad, lenet.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(lenet, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    res = train_network(input_data, input_label)

    print("RUNTIME_COMPILE", res, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", res.asnumpy().shape, "RUNTIME_CACHE")
