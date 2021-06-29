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
common architecture.
"""
import mindspore.nn as nn
from src.utils.cunstom_op import GlobalAvgPooling

__all__ = ['CommonHead']

class CommonHead(nn.Cell):
    """
    common architecture definition.

    Args:
        num_classes (int): Number of classes.
        out_channels (int): Output channels.

    Returns:
        Tensor, output tensor.
    """
    def __init__(self, num_classes, out_channels):
        super(CommonHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = nn.Dense(out_channels, num_classes, has_bias=True).add_flags_recursive(fp16=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        return x
