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
This file contains TextFileDataset for loading text files as a dataset.
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""
import mindspore._c_dataengine as cde

from .samplers import Shuffle
from .datasets import TextBaseDataset, SourceDataset
from .validators import check_textfiledataset



class TextFileDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses datasets stored on disk in text format.
    The generated dataset has one column :py:obj:`[text]` with type string.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None`` , will include all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers (8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in.
            Default: ``Shuffle.GLOBAL`` .
            If `shuffle` is ``False`` , no shuffling will be performed.
            If `shuffle` is ``True`` , performs global shuffle.
            There are two levels of shuffling, desired shuffle enum defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.GLOBAL`` : Shuffle both the files and samples, same as setting shuffle to True.
            - ``Shuffle.FILES`` : Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            When this argument is specified, `num_samples` reflects the maximum sample number per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None`` , which means no cache is used.

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_

    Examples:
        >>> import mindspore.dataset as ds
        >>> text_file_list = ["/path/to/text_file_dataset_file"] # contains 1 or multiple text files
        >>> dataset = ds.TextFileDataset(dataset_files=text_file_list)
    """

    @check_textfiledataset
    def __init__(self, dataset_files, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

    def parse(self, children=None):
        return cde.TextFileNode(self.dataset_files, self.num_samples, self.shuffle_flag, self.num_shards,
                                self.shard_id)
