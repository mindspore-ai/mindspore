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

from blocks import ShuffleV2Block

from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.operations as P


class ShuffleNetV2(nn.Cell):
    def __init__(self, input_size=224, n_class=1000, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(num_features=input_channel, momentum=0.9),
            nn.ReLU(),
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.features = nn.SequentialCell([*self.features])

        self.conv_last = nn.SequentialCell([
            nn.Conv2d(in_channels=input_channel, out_channels=self.stage_out_channels[-1], kernel_size=1, stride=1,
                      pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=self.stage_out_channels[-1], momentum=0.9),
            nn.ReLU()
        ])
        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=7, pad_mode='valid')
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(keep_prob=0.8)
        self.classifier = nn.SequentialCell([nn.Dense(in_channels=self.stage_out_channels[-1],
                                                      out_channels=n_class, has_bias=False)])
        ##TODO init weights
        self._initialize_weights()

    def construct(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = P.Reshape()(x, (-1, self.stage_out_channels[-1],))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    m.weight.set_parameter_data(Tensor(np.random.normal(0, 0.01,
                                                                        m.weight.data.shape).astype("float32")))
                else:
                    m.weight.set_parameter_data(Tensor(np.random.normal(0, 1.0/m.weight.data.shape[1],
                                                                        m.weight.data.shape).astype("float32")))

            if isinstance(m, nn.Dense):
                m.weight.set_parameter_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
