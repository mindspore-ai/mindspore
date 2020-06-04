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
This module provides APIs to load and process various datasets: MNIST,
CIFAR-10, CIFAR-100, VOC, ImageNet, CelebA dataset, etc. It also supports
datasets in special format, including mindrecord, tfrecord, manifest. Users
can also create samplers with this module to sample data.
"""

from .core.configuration import config
from .engine.datasets import TFRecordDataset, ImageFolderDatasetV2, MnistDataset, MindDataset, \
    GeneratorDataset, ManifestDataset, Cifar10Dataset, Cifar100Dataset, VOCDataset, CelebADataset, TextFileDataset, \
    Schema, Shuffle, zip, RandomDataset
from .engine.samplers import DistributedSampler, PKSampler, RandomSampler, SequentialSampler, SubsetRandomSampler, \
    WeightedRandomSampler, Sampler
from .engine.serializer_deserializer import serialize, deserialize, show
from .engine.graphdata import GraphData

__all__ = ["config", "ImageFolderDatasetV2", "MnistDataset",
           "MindDataset", "GeneratorDataset", "TFRecordDataset",
           "ManifestDataset", "Cifar10Dataset", "Cifar100Dataset", "CelebADataset",
           "VOCDataset", "TextFileDataset", "Schema", "DistributedSampler", "PKSampler", "RandomSampler",
           "SequentialSampler", "SubsetRandomSampler", "WeightedRandomSampler", "zip", "GraphData"]
