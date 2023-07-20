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
their own datasets with this module.

Besides, this module provides APIs to sample data while loading.

We can enable cache in most of the dataset with its key arguments 'cache'. Please notice that cache is not supported
on Windows platform yet. Do not use it while loading and processing data on Windows. More introductions and limitations
can refer `Single-Node Tensor Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms
    import mindspore.dataset.vision as vision


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

- Loading datasets: Users can easily load supported datasets using the `*Dataset` class, or load Python layer
  customized datasets through `UDF Loader` + `GeneratorDataset` . At the same time, the loading class method can
  accept a variety of parameters such as sampler, data slicing, and data shuffle;
- Dataset operation: The user uses the dataset object method `.shuffle` / `.filter` / `.skip` / `.split` /
  `.take` / ... to further shuffle, filter, skip, and obtain the maximum number of samples of datasets;
- Dataset sample transform operation: The user can add data transform operations
  ( `vision transform <https://mindspore.cn/docs/en/master/api_python/mindspore.\
  dataset.transforms.html#module-mindspore.dataset.vision>`_ ,
  `NLP transform <https://mindspore.cn/docs/en/master/api_python/mindspore.\
  dataset.transforms.html#module-mindspore.dataset.text>`_ ,
  `audio transform <https://mindspore.cn/docs/en/master/api_python/mindspore.\
  dataset.transforms.html#module-mindspore.dataset.audio>`_ ) to the map
  operation to perform transformations. During data preprocessing, multiple map operations can be defined to
  perform different transform operations to different fields. The data transform operation can also be a
  user-defined transform `pyfunc` (Python function);
- Batch: After the transformation of the samples, the user can use the batch operation to organize multiple samples
  into batches, or use self-defined batch logic with the parameter `per_batch_map` applied;
- Iterator: Finally, the user can use the dataset object method `create_dict_iterator` to create an
  iterator, which can output the preprocessed data cyclically.

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

__all__ = []
__all__.extend(engine.__all__)
