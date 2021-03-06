# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================

"""
Introduction to dataset/engine:

dataset/engine supports various formats of datasets, including ImageNet, TFData,
MNIST, Cifar10/100, Manifest, MindRecord, etc. This module could load data in
high performance and parse data precisely. It also provides the following
operations for users to preprocess data: shuffle, batch, repeat, map, and zip.
"""

from ..core import config
from .cache_client import DatasetCache
from .datasets import *
from .graphdata import GraphData, SamplingStrategy
from .iterators import *
from .samplers import *
from .serializer_deserializer import compare, deserialize, serialize, show

__all__ = ["CelebADataset", "Cifar100Dataset", "Cifar10Dataset", "CLUEDataset", "CocoDataset", "CSVDataset",
           "GeneratorDataset", "GraphData", "ImageFolderDataset", "ManifestDataset", "MindDataset", "MnistDataset",
           "NumpySlicesDataset", "PaddedDataset", "TextFileDataset", "TFRecordDataset", "VOCDataset",
           "DistributedSampler", "PKSampler", "RandomSampler", "SequentialSampler", "SubsetRandomSampler",
           "WeightedRandomSampler", "SubsetSampler",
           "config", "DatasetCache", "Schema", "zip"]
