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
At the heart of MindSpore data loading utility is the `mindspore.dataset` module.
It is a `dataset engine <https://www.mindspore.cn/docs/en/master/features/data_engine.html>`_ based on pipline design.

This module provides the following data loading methods to help users load datasets into MindSpore.

- User defined dataset loading: allows users to define `Random-accessible(Map-style) dataset
  <https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#random-accessible-dataset>`_ or
  `Iterable-style dataset <https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#iterable-dataset>`_
  to customize data reading and processing logic.
- Standard format dataset loading: support loading dataset files in standard data formats, including
  `MindRecord <https://www.mindspore.cn/tutorials/en/master/dataset/record.html>`_,
  `TFRecord <https://tensorflow.google.cn/tutorials/load_data/tfrecord.md?hl=en>`_ .
- Open source dataset loading: supports reading `open source datasets <#open-source>`_ ,
  such as MNIST, CIFAR-10, CLUE, LJSpeech, etc.

In addition, this module also provides data sampler, transformations, batching, as well as basic configurations
such as random seed, parallelism setting and other features, to be used in conjunction with the dataset loading.

- Data Sampler: Provides various common `sampler <#sampler-1>`_, such as RandomSampler, DistributedSampler, etc.
- Data Transformations: Provides multiple `dataset operations <https://www.mindspore.cn/docs/en/master/api_python/
  dataset/mindspore.dataset.GeneratorDataset.html#pre-processing-operation>`_ to perform data augmentation, batching.
- Basic Configuration: Provides `pipeline configuration <#config>`_ for random seed setting, parallelism setting,
  data recovery mode, etc.

Descriptions of common dataset terms are as follows:

- Dataset, the base class of all the datasets. It provides data processing methods to help preprocess the data.
- SourceDataset, an abstract class to represent the source of dataset pipeline which produces data from data
  sources such as files and databases.
- MappableDataset, an abstract class to represent a source dataset which supports for random access.
- Iterator, the base class of dataset iterator for enumerating elements.

Introduction to data processing pipeline
----------------------------------------

.. image:: dataset_pipeline_en.png

As shown in the above figure, the mindspore dataset module makes it easy for users to define data preprocessing
pipelines and transform samples in the dataset in the most efficient (multi-process / multi-thread) manner.
The specific steps are as follows:

- Loading datasets: Users can easily load supported datasets using the Dataset class
  (`Standard-format Dataset
  <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#standard-format>`_,
  `Vision Dataset <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#vision>`_,
  `NLP Dataset <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#text>`_,
  `Audio Dataset <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#audio>`_,
  or load Python layer customized datasets through
  `User Defined Dataset
  <https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#user-defined>`_,
- Dataset operation: The user uses the dataset object method
  `.shuffle <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/
  mindspore.dataset.Dataset.shuffle.html#mindspore.dataset.Dataset.shuffle>`_ /
  `.filter <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/
  mindspore.dataset.Dataset.filter.html#mindspore.dataset.Dataset.filter>`_ /
  `.skip <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/
  mindspore.dataset.Dataset.skip.html#mindspore.dataset.Dataset.skip>`_ /
  `.split <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/
  mindspore.dataset.Dataset.split.html#mindspore.dataset.Dataset.split>`_ /
  `.take <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/
  mindspore.dataset.Dataset.take.html#mindspore.dataset.Dataset.take>`_ /
  … to further shuffle, filter, skip, and obtain the maximum number of samples of datasets.
- Dataset sample transform operation: The user can add data transform operations
  (`vision transform <https://www.mindspore.cn/docs/en/master/api_python/
  mindspore.dataset.transforms.html#module-mindspore.dataset.vision>`_,
  `nlp transform <https://www.mindspore.cn/docs/en/master/api_python/
  mindspore.dataset.transforms.html#module-mindspore.dataset.text>`_,
  `audio transform <https://www.mindspore.cn/docs/en/master/api_python/
  mindspore.dataset.transforms.html#module-mindspore.dataset.audio>`_ )
  to the `.map <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/
  mindspore.dataset.Dataset.map.html#mindspore.dataset.Dataset.map>`_ operation to perform transforms.
  During data preprocessing, multiple map operations can be defined
  to perform different transform operations to different fields.
  The data transform operation can also be a user-defined Python function.
- Batch: After the transforms of the samples, the user can use the
  `.batch <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/batch/
  mindspore.dataset.Dataset.batch.html#mindspore.dataset.Dataset.batch>`_ operation to organize multiple samples
  into batches, or use self-defined batch logic with the parameter per_batch_map applied.
- Iterator: Finally, user can use the method
  `.create_dict_iterator <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/iterator/
  mindspore.dataset.Dataset.create_dict_iterator.html#mindspore.dataset.Dataset.create_dict_iterator>`_ or
  `.create_tuple_iterator <https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/iterator/
  mindspore.dataset.Dataset.create_tuple_iterator.html#mindspore.dataset.Dataset.create_tuple_iterator>`_ to create
  an iterator, which can output the preprocessed data cyclically.

Quick start of Dataset Pipeline
-------------------------------

For a quick start of using Dataset Pipeline, download `Load & Process Data With Dataset Pipeline
<https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_
to local and run in sequence.

"""

from .core import config
from .engine import *
from .engine.cache_client import DatasetCache
from .engine.datasets import *
from .engine.samplers import *
from .engine.serializer_deserializer import compare, deserialize, serialize, show
from .utils.line_reader import LineReader
from . import dataloader

__all__ = []
__all__.extend(engine.__all__)
