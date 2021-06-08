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
"""Image classification."""
import mindspore.nn as nn
from mindspore.ops import operations as P

from src.cspdarknet53 import cspdarknet53
from src.head import CommonHead
from src.utils.var_init import default_recurisive_init

class ImageClassificationNetwork(nn.Cell):
    """Architecture of Image Classification Network."""
    def __init__(self, backbone, head, include_top=True, activation="None"):
        super(ImageClassificationNetwork, self).__init__()
        self.backbone = backbone
        self.include_top = include_top
        self.need_activation = False
        if self.include_top:
            self.head = head
            if activation != "None":
                self.need_activation = True
                if activation == "Sigmoid":
                    self.activation = P.Sigmoid()
                elif activation == "Softmax":
                    self.activation = P.Softmax()
                else:
                    raise NotImplementedError("The activation {} not in ['Sigmoid', 'Softmax'].".format(activation))

    def construct(self, x):
        x = self.backbone(x)
        if self.include_top:
            x = self.head(x)
            if self.need_activation:
                x = self.activation(x)
        return x

class CSPDarknet53(ImageClassificationNetwork):
    """CSPDarknet53 architecture."""
    def __init__(self, num_classes=1000, include_top=True, activation="None"):
        backbone = cspdarknet53()
        out_channels = backbone.get_out_channels()
        head = CommonHead(num_classes=num_classes, out_channels=out_channels)
        super(CSPDarknet53, self).__init__(backbone, head, include_top, activation)

        default_recurisive_init(self)
