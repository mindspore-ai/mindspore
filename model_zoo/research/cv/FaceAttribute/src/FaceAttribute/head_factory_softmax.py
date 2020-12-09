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
"""Face attribute head."""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn import Cell

from src.FaceAttribute.custom_net import fc_with_initialize

__all__ = ['get_attri_head']


class AttriHead(Cell):
    '''Attribute Head.'''
    def __init__(self, flat_dim, fc_dim, attri_num_list):
        super(AttriHead, self).__init__()
        self.fc1 = fc_with_initialize(flat_dim, fc_dim)
        self.fc1_relu = P.ReLU()
        self.fc1_bn = nn.BatchNorm1d(fc_dim, affine=False)
        self.attri_fc1 = fc_with_initialize(fc_dim, attri_num_list[0])
        self.attri_fc1_relu = P.ReLU()
        self.attri_bn1 = nn.BatchNorm1d(attri_num_list[0], affine=False)
        self.softmax1 = P.Softmax()

        self.fc2 = fc_with_initialize(flat_dim, fc_dim)
        self.fc2_relu = P.ReLU()
        self.fc2_bn = nn.BatchNorm1d(fc_dim, affine=False)
        self.attri_fc2 = fc_with_initialize(fc_dim, attri_num_list[1])
        self.attri_fc2_relu = P.ReLU()
        self.attri_bn2 = nn.BatchNorm1d(attri_num_list[1], affine=False)
        self.softmax2 = P.Softmax()

        self.fc3 = fc_with_initialize(flat_dim, fc_dim)
        self.fc3_relu = P.ReLU()
        self.fc3_bn = nn.BatchNorm1d(fc_dim, affine=False)
        self.attri_fc3 = fc_with_initialize(fc_dim, attri_num_list[2])
        self.attri_fc3_relu = P.ReLU()
        self.attri_bn3 = nn.BatchNorm1d(attri_num_list[2], affine=False)
        self.softmax3 = P.Softmax()

    def construct(self, x):
        '''Construct function.'''
        output0 = self.fc1(x)
        output0 = self.fc1_relu(output0)
        output0 = self.fc1_bn(output0)
        output0 = self.attri_fc1(output0)
        output0 = self.attri_fc1_relu(output0)
        output0 = self.attri_bn1(output0)
        output0 = self.softmax1(output0)

        output1 = self.fc2(x)
        output1 = self.fc2_relu(output1)
        output1 = self.fc2_bn(output1)
        output1 = self.attri_fc2(output1)
        output1 = self.attri_fc2_relu(output1)
        output1 = self.attri_bn2(output1)
        output1 = self.softmax2(output1)

        output2 = self.fc3(x)
        output2 = self.fc3_relu(output2)
        output2 = self.fc3_bn(output2)
        output2 = self.attri_fc3(output2)
        output2 = self.attri_fc3_relu(output2)
        output2 = self.attri_bn3(output2)
        output2 = self.softmax3(output2)

        return output0, output1, output2


def get_attri_head(flat_dim, fc_dim, attri_num_list):
    attri_head = AttriHead(flat_dim, fc_dim, attri_num_list)
    return attri_head
