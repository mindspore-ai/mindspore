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
# ===========================================================================
"""DSCNN network."""
import math

import mindspore.nn as nn


class DSCNN(nn.Cell):
    '''Build DSCNN network.'''
    def __init__(self, model_settings, model_size_info):
        super(DSCNN, self).__init__()
        # N C H W
        label_count = model_settings['label_count']
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        t_dim = input_time_size
        f_dim = input_frequency_size
        num_layers = model_size_info[0]
        conv_feat = [None] * num_layers
        conv_kt = [None] * num_layers
        conv_kf = [None] * num_layers
        conv_st = [None] * num_layers
        conv_sf = [None] * num_layers
        i = 1
        for layer_no in range(0, num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1
        seq_cell = []
        in_channel = 1
        for layer_no in range(0, num_layers):
            if layer_no == 0:
                seq_cell.append(nn.Conv2d(in_channels=in_channel, out_channels=conv_feat[layer_no],
                                          kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                          stride=(conv_st[layer_no], conv_sf[layer_no]),
                                          pad_mode="same", padding=0, has_bias=False))
                seq_cell.append(nn.BatchNorm2d(num_features=conv_feat[layer_no], momentum=0.98))
                in_channel = conv_feat[layer_no]
            else:
                seq_cell.append(nn.Conv2d(in_channel, in_channel, kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                          stride=(conv_st[layer_no], conv_sf[layer_no]), pad_mode='same',
                                          has_bias=False, group=in_channel, weight_init='ones'))
                seq_cell.append(nn.BatchNorm2d(num_features=in_channel, momentum=0.98))
                seq_cell.append(nn.ReLU())
                seq_cell.append(nn.Conv2d(in_channels=in_channel, out_channels=conv_feat[layer_no], kernel_size=(1, 1),
                                          pad_mode="same"))
                seq_cell.append(nn.BatchNorm2d(num_features=conv_feat[layer_no], momentum=0.98))
                seq_cell.append(nn.ReLU())
                in_channel = conv_feat[layer_no]
            t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
            f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))
        seq_cell.append(nn.AvgPool2d(kernel_size=(t_dim, f_dim)))  # to fix ?
        seq_cell.append(nn.Flatten())
        seq_cell.append(nn.Dropout(model_settings['dropout1']))
        seq_cell.append(nn.Dense(in_channel, label_count))
        self.model = nn.SequentialCell(seq_cell)

    def construct(self, x):
        x = self.model(x)
        return x
