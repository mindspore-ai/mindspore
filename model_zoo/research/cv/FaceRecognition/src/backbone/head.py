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
"""Face Recognition head."""
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore.ops import operations as P
from src.custom_net import Cut, bn_with_initialize, fc_with_initialize

__all__ = ['get_head']


class Head0(Cell):
    '''Head0'''
    def __init__(self, emb_size, args=None):
        super(Head0, self).__init__()
        if args.pre_bn == 1:
            self.bn1 = bn_with_initialize(512, use_inference=args.inference)
        else:
            self.bn1 = Cut()

        if args is not None:
            if args.use_drop == 1:
                self.drop = nn.Dropout(keep_prob=0.4)
            else:
                self.drop = Cut()
        else:
            self.drop = nn.Dropout(keep_prob=0.4)

        self.fc1 = fc_with_initialize(512 * 7 * 7, emb_size)
        if args.inference == 1:
            self.bn2 = Cut()
        else:
            self.bn2 = nn.BatchNorm1d(emb_size, affine=False, momentum=0.9).add_flags_recursive(fp32=True)
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        x = self.bn1(x)
        x = self.drop(x)
        b, _, _, _ = self.shape(x)
        shp = (b, -1)
        x = self.reshape(x, shp)
        x = self.fc1(x)
        x = self.bn2(x)

        return x


def get_head(args):
    emb_size = args.emb_size
    return Head0(emb_size, args)
