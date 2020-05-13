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
"""
test pooling api
"""
import mindspore.nn as nn


class MaxNet(nn.Cell):
    """MaxNet definition"""

    def __init__(self,
                 kernel_size,
                 stride=None):
        super(MaxNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size,
                                    stride)

    def construct(self, input_x):
        return self.maxpool(input_x)


class AvgNet(nn.Cell):
    def __init__(self,
                 kernel_size,
                 stride=None):
        super(AvgNet, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size,
                                    stride)

    def construct(self, input_x):
        return self.avgpool(input_x)
