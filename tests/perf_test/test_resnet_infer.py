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

import numpy as np

from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
from .resnet_example import resnet50


def test_compile():
    net = resnet50()
    inp = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    _cell_graph_executor.compile(net, inp)
