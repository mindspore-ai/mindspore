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

"""Resnet test."""

# pylint: disable=unused-wildcard-import, wildcard-import

import numpy as np

from mindspore import Tensor
from .resnet_example import resnet50
from ..train_step_wrap import train_step_without_opt


def test_resnet50_pynative():
    net = train_step_without_opt(resnet50())
    inp = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net(inp, label)
