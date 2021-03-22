# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from src.unet3d_parts import Down, Up

class UNet3d(nn.Cell):
    def __init__(self, config=None):
        super(UNet3d, self).__init__()
        self.n_channels = config.in_channels
        self.n_classes = config.num_classes

        # down
        self.transpose = P.Transpose()
        self.down1 = Down(in_channel=self.n_channels, out_channel=16, dtype=mstype.float16).to_float(mstype.float16)
        self.down2 = Down(in_channel=16, out_channel=32, dtype=mstype.float16).to_float(mstype.float16)
        self.down3 = Down(in_channel=32, out_channel=64, dtype=mstype.float16).to_float(mstype.float16)
        self.down4 = Down(in_channel=64, out_channel=128, dtype=mstype.float16).to_float(mstype.float16)
        self.down5 = Down(in_channel=128, out_channel=256, stride=1, kernel_size=(1, 1, 1), \
                          dtype=mstype.float16).to_float(mstype.float16)

        # up
        self.up1 = Up(in_channel=256, down_in_channel=128, out_channel=64, \
                      dtype=mstype.float16).to_float(mstype.float16)
        self.up2 = Up(in_channel=64, down_in_channel=64, out_channel=32, \
                      dtype=mstype.float16).to_float(mstype.float16)
        self.up3 = Up(in_channel=32, down_in_channel=32, out_channel=16, \
                      dtype=mstype.float16).to_float(mstype.float16)
        self.up4 = Up(in_channel=16, down_in_channel=16, out_channel=self.n_classes, \
                      dtype=mstype.float16, is_output=True).to_float(mstype.float16)

        self.cast = P.Cast()


    def construct(self, input_data):
        input_data = self.cast(input_data, mstype.float16)
        x1 = self.down1(input_data)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.cast(x, mstype.float32)
        return x
