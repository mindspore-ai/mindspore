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

from ..callback import DSCallback, WaitedDSCallback
from ..core import config
from .cache_client import DatasetCache
from .datasets import *
from .datasets_vision import *
from .datasets_text import *
from .datasets_audio import *
from .datasets_standard_format import *
from .datasets_user_defined import *
from .graphdata import GraphData, SamplingStrategy, OutputFormat
from .iterators import *
from .samplers import *
from .serializer_deserializer import compare, deserialize, serialize, show

__all__ = ["Caltech101Dataset",        # vision dataset
           "Caltech256Dataset",        # vision dataset
           "CelebADataset",            # vision dataset
           "Cifar10Dataset",           # vision dataset
           "Cifar100Dataset",          # vision dataset
           "CityscapesDataset",        # vision dataset
           "CocoDataset",              # vision dataset
           "DIV2KDataset",             # vision dataset
           "EMnistDataset",            # vision dataset
           "FakeImageDataset",         # vision dataset
           "FashionMnistDataset",      # vision dataset
           "FlickrDataset",            # vision dataset
           "Flowers102Dataset",        # vision dataset
           "ImageFolderDataset",       # vision dataset
           "KMnistDataset",            # vision dataset
           "ManifestDataset",          # vision dataset
           "MnistDataset",             # vision dataset
           "PhotoTourDataset",         # vision dataset
           "Places365Dataset",         # vision dataset
           "QMnistDataset",            # vision dataset
           "RandomDataset",            # vision dataset
           "SBDataset",                # vision dataset
           "SBUDataset",               # vision dataset
           "SemeionDataset",           # vision dataset
           "STL10Dataset",             # vision dataset
           "SVHNDataset",              # vision dataset
           "USPSDataset",              # vision dataset
           "VOCDataset",               # vision dataset
           "WIDERFaceDataset",         # vision dataset
           "AGNewsDataset",            # text dataset
           "AmazonReviewDataset",      # text dataset
           "CLUEDataset",              # text dataset
           "CoNLL2000Dataset",         # text dataset
           "CSVDataset",               # text dataset
           "DBpediaDataset",           # text dataset
           "EnWik9Dataset",            # text dataset
           "IMDBDataset",              # text dataset
           "IWSLT2016Dataset",         # text dataset
           "IWSLT2017Dataset",         # text dataset
           "PennTreebankDataset",      # text dataset
           "SogouNewsDataset",         # text dataset
           "TextFileDataset",          # text dataset
           "UDPOSDataset",             # text dataset
           "WikiTextDataset",          # text dataset
           "YahooAnswersDataset",      # text dataset
           "YelpReviewDataset",        # text dataset
           "LJSpeechDataset",          # audio dataset
           "SpeechCommandsDataset",    # audio dataset
           "TedliumDataset",           # audio dataset
           "YesNoDataset",             # audio dataset
           "MindDataset",              # standard format dataset
           "TFRecordDataset",          # standard format dataset
           "GeneratorDataset",         # user defined dataset
           "NumpySlicesDataset",       # user defined dataset
           "PaddedDataset",            # user defined dataset
           "GraphData",                # graph data
           "DistributedSampler",       # sampler
           "RandomSampler",            # sampler
           "SequentialSampler",        # sampler
           "SubsetRandomSampler",      # sampler
           "SubsetSampler",            # sampler
           "PKSampler",                # sampler
           "WeightedRandomSampler",    # sampler
           "DatasetCache",
           "DSCallback",
           "WaitedDSCallback",
           "Schema",
           "compare",
           "deserialize",
           "serialize",
           "show",
           "zip"]
