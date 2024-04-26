# Copyright 2024 Huawei Technologies Co., Ltd
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
nn Extend.
"""
from __future__ import absolute_import

from mindspore.nn.extend.embedding import Embedding
from mindspore.nn.extend.basic import Linear
from mindspore.nn.extend.pooling import MaxPool2d
from mindspore.nn.extend import layer
from mindspore.nn.extend.layer import *

__all__ = ['Embedding', 'Linear', 'MaxPool2d']
__all__.extend(layer.__all__)

__all__.sort()
