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
"""model define"""
import math
from mindspore import nn
import mindspore.ops.operations as P
from mindspore.common import initializer as init


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(
                    init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(
                    init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(init.initializer(
                    init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(
                    init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.GroupNorm):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class Generator(nn.Cell):
    """Generator"""
    def __init__(self, input_dim, output_dim=1, input_size=28, class_num=10):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num
        self.concat = P.Concat(1)

        self.fc = nn.SequentialCell(
            nn.Dense(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dense(1024, 128 * (self.input_size // 4)
                     * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4)
                           * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.SequentialCell(
            nn.Conv2dTranspose(128, 64, 4, 2, padding=0,
                               has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2dTranspose(64, self.output_dim, 4, 2,
                               padding=0, has_bias=True, pad_mode='same'),
            nn.Tanh(),
        )
        init_weights(self.deconv, 'KaimingUniform', math.sqrt(5))

    def construct(self, input_param, label):
        """construct"""
        x = self.concat((input_param, label))
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class Discriminator(nn.Cell):
    """Discriminator"""
    def __init__(self, batch_size, input_dim=1, output_dim=1, input_size=28, class_num=10):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num
        self.concat = P.Concat(1)
        self.ExpandDims = P.ExpandDims()
        self.expand = P.BroadcastTo

        self.conv = nn.SequentialCell(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.SequentialCell(
            nn.Dense(128 * (self.input_size // 4) *
                     (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, self.output_dim),
            nn.Sigmoid(),
        )
        init_weights(self.conv, 'KaimingUniform', math.sqrt(5))

    def construct(self, input_param, label):
        """construct"""
        # expand_fill
        label_fill = self.ExpandDims(label, 2)
        label_fill = self.ExpandDims(label_fill, 3)
        shape = (self.batch_size, 10, self.input_size, self.input_size)
        label_fill = self.expand(shape)(label_fill)

        # forward
        x = self.concat((input_param, label_fill))
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x
