# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
This file contains specific vision dataset loading classes. You can easily use
these classes to load the prepared dataset. For example:
    ImageFolderDataset: which is about ImageNet dataset.
    Cifar10Dataset: which is cifar10 binary version dataset.
    Cifar100Dataset: which is cifar100 binary version dataset.
    MnistDataset: which is mnist dataset.
    ...
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""

import mindspore._c_dataengine as cde

from .datasets import VisionBaseDataset, SourceDataset, MappableDataset, Schema
from .validators import check_imagefolderdataset, check_manifestdataset, check_random_dataset

from ..core.validator_helpers import replace_none


class ImageFolderDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads images from a tree of directories.
    All images within one folder have the same label.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: ``None`` , all images.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers (8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: ``None`` , expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: ``None`` , expected order behavior shown in the table below.
        extensions (list[str], optional): List of file extensions to be
            included in the dataset. Default: ``None`` .
        class_indexing (dict, optional): A str-to-int mapping from folder name to index.
            Default: ``None`` , the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0.
        decode (bool, optional): Decode the images after reading. Default: ``False``.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: ``None`` . When this argument is specified, `num_samples` reflects
            the maximum sample number per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None`` , which means no cache is used.
        decrypt (callable, optional): Image decryption function, which accepts the path of the encrypted image file
            and returns the decrypted bytes data. Default: ``None`` , no decryption.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        RuntimeError: If `class_indexing` is not a dictionary.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_

    Note:
        - The shape of the image column is [undecoded_image_size] if `decode` flag is ``False``, or [H,W,C] otherwise.
        - The parameters `num_samples` , `shuffle` , `num_shards` , `shard_id` can be used to control the sampler
          used in the dataset, and their effects when combined with parameter `sampler` are as follows.

    .. include:: mindspore.dataset.sampler.txt

    Examples:
        >>> import mindspore.dataset as ds
        >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in image_folder_dataset_dir with 8 threads
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 num_parallel_workers=8)
        >>>
        >>> # 2) Read all samples (image files) from folder cat and folder dog with label 0 and 1
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 class_indexing={"cat":0, "dog":1})
        >>>
        >>> # 3) Read all samples (image files) in image_folder_dataset_dir with extensions .JPEG
        >>> #    and .png (case sensitive)
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 extensions=[".JPEG", ".png"])

    About ImageFolderDataset:

    You can construct the following directory structure from your dataset files and read by MindSpore's API.

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── class1
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class2
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class3
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── classN
             ├── ...
    """

    @check_imagefolderdataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 extensions=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None,
                 decrypt=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.extensions = replace_none(extensions, [])
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)
        self.decrypt = decrypt

    def parse(self, children=None):
        return cde.ImageFolderNode(self.dataset_dir, self.decode, self.sampler, self.extensions, self.class_indexing,
                                   self.decrypt)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> import mindspore.dataset as ds
            >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
            >>>
            >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir)
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.class_indexing is None or not self.class_indexing:
            runtime_getter = self._init_tree_getters()
            _class_indexing = runtime_getter[0].GetClassIndexing()
            for pair in _class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class ManifestDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset for reading images from a Manifest file.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of uint64 type.

    Args:
        dataset_file (str): File to be read.
        usage (str, optional): Acceptable usages include ``'train'``, ``'eval'`` and ``'inference'``.
            Default: ``'train'``.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: ``None`` , will include all images.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers (8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: ``None`` , expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: ``None`` , expected order behavior shown in the table below.
        class_indexing (dict, optional): A str-to-int mapping from label name to index.
            Default: ``None`` , the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0.
        decode (bool, optional): Decode the images after reading. Default: ``False``.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into during distributed training. Default: ``None`` .
            When this argument is specified, `num_samples` reflects
            the max number of samples per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None`` , which means no cache is used.

    Raises:
        RuntimeError: If `dataset_file` are not valid or do not exist.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_

    Note:
        - If `decode` is ``False`` , the "image" column will get the 1D raw bytes of the image.
          Otherwise, a decoded image with shape :math:`[H,W,C]` will be returned.
        - The parameters `num_samples` , `shuffle` , `num_shards` , `shard_id` can be used to control the sampler
          used in the dataset, and their effects when combined with parameter `sampler` are as follows.

    .. include:: mindspore.dataset.sampler.txt

    Examples:
        >>> import mindspore.dataset as ds
        >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
        >>>
        >>> # 1) Read all samples specified in manifest_dataset_dir dataset with 8 threads for training
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, usage="train", num_parallel_workers=8)
        >>>
        >>> # 2) Read samples (specified in manifest_file.manifest) for shard 0 in a 2-way distributed training setup
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, num_shards=2, shard_id=0)

    About Manifest dataset:

    Manifest file contains a list of files included in a dataset, including basic file info such as File name and File
    ID, along with extended file metadata. Manifest is a data format file supported by Huawei Modelarts. For details,
    see `Specifications for Importing the Manifest File
    <https://support.huaweicloud.com/intl/en-us/sdkreference-modelarts/modelarts_04_0349.html>`_ .

    .. code-block::

        .
        └── manifest_dataset_directory
            ├── train
            │    ├── 1.JPEG
            │    ├── 2.JPEG
            │    ├── ...
            ├── eval
            │    ├── 1.JPEG
            │    ├── 2.JPEG
            │    ├── ...
    """

    @check_manifestdataset
    def __init__(self, dataset_file, usage="train", num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_file = dataset_file
        self.decode = replace_none(decode, False)
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})

    def parse(self, children=None):
        return cde.ManifestNode(self.dataset_file, self.usage, self.sampler, self.class_indexing, self.decode)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> import mindspore.dataset as ds
            >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
            >>>
            >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir)
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class RandomDataset(SourceDataset, VisionBaseDataset):
    """
    A source dataset that generates random data.

    Args:
        total_rows (int, optional): Number of samples for the dataset to generate.
            Default: ``None`` , number of samples is random.
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by :class:`mindspore.dataset.Schema` are
            acceptable. Default: ``None`` .
        columns_list (list[str], optional): List of column names of the dataset.
            Default: ``None`` , the columns will be named like this "c0", "c1", "c2" etc.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None`` , all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers (8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None`` , which means no cache is used.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: ``None`` , expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into during distributed training. Default: ``None`` .
            When this argument is specified, `num_samples` reflects
            the maximum sample number per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.

    Raises:
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        TypeError: If `total_rows` is not of type int.
        TypeError: If `num_shards` is not of type int.
        TypeError: If `num_parallel_workers` is not of type int.
        TypeError: If `shuffle` is not of type bool.
        TypeError: If `columns_list` is not of type list.

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_

    Examples:
        >>> from mindspore import dtype as mstype
        >>> import mindspore.dataset as ds
        >>>
        >>> schema = ds.Schema()
        >>> schema.add_column('image', de_type=mstype.uint8, shape=[2])
        >>> schema.add_column('label', de_type=mstype.uint8, shape=[1])
        >>> # apply dataset operations
        >>> ds1 = ds.RandomDataset(schema=schema, total_rows=50, num_parallel_workers=4)
    """

    @check_random_dataset
    def __init__(self, total_rows=None, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 cache=None, shuffle=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.total_rows = replace_none(total_rows, 0)
        if self.total_rows != 0 and self.num_samples != 0:
            self.total_rows = min(self.total_rows, self.num_samples)
        if self.num_samples != 0:
            self.total_rows = self.num_samples
        if schema is not None:
            self.total_rows = replace_none(total_rows, Schema.get_num_rows(schema))
        self.schema = replace_none(schema, "")
        self.columns_list = replace_none(columns_list, [])

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.RandomNode(self.total_rows, schema, self.columns_list)
