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
from ...father_net.FatherNet import FatherNet, NetBUtImport

class SubNetUtImport(FatherNet):
    def __init__(self, in_channels=32, **kwargs):
        super(SubNetUtImport, self).__init__(**kwargs)
        self.dense = nn.Dense(in_channels=in_channels, out_channels=32, has_bias=False, weight_init="ones")
        self.netb = NetBUtImport()

    def construct(self, x):
        x = self.dense(x)
        x = self.netb(x)
        return x
