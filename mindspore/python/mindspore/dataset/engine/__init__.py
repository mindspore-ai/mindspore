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
from .datasets_standard_format import *
from .datasets_user_defined import *
from .iterators import *
from .obs.obs_mindrecord_dataset import *
from .samplers import *

__all__ = [
           "ImageFolderDataset",       # Vision
           "ManifestDataset",          # Vision
           "RandomDataset",            # Vision
           "AGNewsDataset",            # Text
           "AmazonReviewDataset",      # Text
           "CLUEDataset",              # Text
           "CoNLL2000Dataset",         # Text
           "DBpediaDataset",           # Text
           "EnWik9Dataset",            # Text
           "IMDBDataset",              # Text
           "IWSLT2016Dataset",         # Text
           "IWSLT2017Dataset",         # Text
           "Multi30kDataset",          # Text
           "PennTreebankDataset",      # Text
           "SogouNewsDataset",         # Text
           "SQuADDataset",             # Text
           "SST2Dataset",              # Text
           "TextFileDataset",          # Text
           "UDPOSDataset",             # Text
           "WikiTextDataset",          # Text
           "YahooAnswersDataset",      # Text
           "YelpReviewDataset",        # Text
           "CSVDataset",               # Standard Format
           "MindDataset",              # Standard Format
           "OBSMindDataset",           # Standard Format
           "TFRecordDataset",          # Standard Format
           "GeneratorDataset",         # User Defined
           "NumpySlicesDataset",       # User Defined
           "PaddedDataset",            # User Defined
           "DistributedSampler",       # Sampler
           "RandomSampler",            # Sampler
           "SequentialSampler",        # Sampler
           "DatasetCache",
           "DSCallback",
           "WaitedDSCallback",
           "Schema",
           "sync_wait_for_dataset",
           "zip"]
