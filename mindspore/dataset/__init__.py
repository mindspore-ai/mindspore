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
"""
This module provides APIs to load and process various common datasets such as MNIST,
CIFAR-10, CIFAR-100, VOC, COCO, ImageNet, CelebA, CLUE, etc. It also supports datasets
in standard format, including MindRecord, TFRecord, Manifest, etc. Users can also define
their owndatasets with this module.

Besides, this module provides APIs to sample data while loading.

Please notice that cache is not supported on Windows platform yet. Please do not use it
while loading and processing data on Windows.
"""

from .core import config
from .engine import *
from .engine.cache_client import DatasetCache
from .engine.datasets import *
from .engine.graphdata import GraphData
from .engine.samplers import *
from .engine.serializer_deserializer import compare, deserialize, serialize, show

__all__ = []
__all__.extend(engine.__all__)
