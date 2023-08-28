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
import mindspore.nn as nn

class FatherNet(nn.Cell):
    def __init__(self, in_channels=32, **kwargs):
        super(FatherNet, self).__init__(**kwargs)
        self.dense = nn.Dense(in_channels=in_channels, out_channels=32, weight_init="ones")

    def construct(self, x):
        x = self.dense(x)
        return x

class NetBUtImport(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, stride=1, weight_init="ones")

    def construct(self, x):
        x = self.conv1(x)
        return x
