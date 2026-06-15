# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
The sampler module provides several samplers to generate data from datasets.
The provided samplers include: DistributedSampler, PKSampler, RandomSampler,
SequentialSampler, SubsetRandomSampler, and WeightedRandomSampler.
Users can also define a custom sampler by extending from the Sampler class.
"""

import copy
import numbers

from enum import Enum
import numpy as np
import mindspore._c_dataengine as cde
import mindspore.dataset as ds
from ..core import validator_helpers as validator


class Shuffle(str, Enum):
    """Specify the shuffle mode.

    - ``Shuffle.FALSE`` : Disable the shuffle.
    - ``Shuffle.ADAPTIVE`` : When the number of dataset samples is less than or equal to 100 million,
      global shuffle is used. When the number of dataset samples is greater than 100 million,
      partial shuffle is used. Shuffle data every 1 million samples.
    - ``Shuffle.GLOBAL`` : Shuffle both the files and samples.
    - ``Shuffle.PARTIAL`` : Shuffle data every 1 million samples.
    - ``Shuffle.FILES`` : Shuffle files only, not shuffle the data within the files.
    - ``Shuffle.INFILE`` : Preserve the order in which the files are read, but shuffle the data within each file.
    """
    FALSE: str = "false"
    ADAPTIVE: str = "adaptive"
    GLOBAL: str = "global"
    PARTIAL: str = "partial"
    FILES: str = "files"
    INFILE: str = "infile"


ShuffleToShuffleMode = {Shuffle.FALSE: cde.ShuffleMode.FALSE,
                        Shuffle.ADAPTIVE: cde.ShuffleMode.ADAPTIVE,
                        Shuffle.GLOBAL: cde.ShuffleMode.GLOBAL,
                        Shuffle.PARTIAL: cde.ShuffleMode.PARTIAL,
                        Shuffle.FILES: cde.ShuffleMode.FILES,
                        Shuffle.INFILE: cde.ShuffleMode.INFILE}


def shuffle_to_shuffle_mode(shuffle):
    """
    Shuffle Enum to Shuffle Mode

    Args:
        shuffle (Shuffle): shuffle flag to shuffle mode in C layer

    Returns:
        ShuffleMode, shuffle mode
    """
    shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
    if not isinstance(shuffle, Shuffle):
        if shuffle is None or shuffle:
            shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
        else:
            shuffle_mode = cde.ShuffleMode.FALSE  # No shuffle
    else:
        shuffle_mode = ShuffleToShuffleMode[shuffle]
    return shuffle_mode


def select_sampler(num_samples, input_sampler, shuffle, num_shards, shard_id):
    """
    Create sampler based on user input.

    Args:
        num_samples (int): Number of samples.
        input_sampler (Union[Iterable, Sampler]): Sampler from user.
        shuffle (Shuffle): Shuffle is FALSE / ADAPTIVE / GLOBAL / PARTIAL / FILES / INFILE
        num_shards (int): Number of shard for sharding.
        shard_id (int): Shard ID.

    Returns:
        Sampler, sampler selected based on user input.
    """
    if input_sampler is None and shuffle not in (Shuffle.FALSE, Shuffle.ADAPTIVE, Shuffle.GLOBAL, Shuffle.PARTIAL,
                                                 Shuffle.FILES, Shuffle.INFILE):
        raise RuntimeError("The input parameter shuffle: {} is not valid.".format(shuffle))

    if input_sampler is not None:
        # If the user provided a sampler, then it doesn't matter what the other args are because
        # we are being asked specifically to use the given sampler.
        # That means the following arguments: num_shards, shard_id, shuffle, num_samples should all
        # be None. Consider this example:
        #     sampler = ds.DistributedSampler(num_shards=8, shard_id=3, shuffle=shuffle)
        #     data1 = ds.VOCDataset(voc_dir, decode=True, sampler=sampler, num_shards=4, shard_id=1)
        # In this case, the user has given different sample-related arguments that contradict each other.
        # To prevent this, only allow the user to manually specify the sampler if those arguments are all None
        if (isinstance(input_sampler, BuiltinSampler) and
                (any(arg is not None for arg in [num_shards, shard_id, shuffle, num_samples]))):
            raise ValueError(
                'Conflicting arguments during sampler assignments. num_samples: {}, num_shards: {},'
                ' shard_id: {}, shuffle: {}.'.format(num_samples, num_shards, shard_id, shuffle))
        if isinstance(input_sampler, BuiltinSampler):
            return input_sampler
        if not isinstance(input_sampler, str) and isinstance(input_sampler, (np.ndarray, list, tuple)):
            return SubsetSampler(input_sampler, num_samples)
        if not isinstance(input_sampler, str) and validator.is_iterable(input_sampler):
            # in this case, the user passed in their own sampler object that's not of type BuiltinSampler
            return IterSampler(input_sampler, num_samples)
        if isinstance(input_sampler, int):
            return SubsetSampler([input_sampler])
        raise TypeError('Unsupported sampler object of type ({})'.format(type(input_sampler)))
    if shuffle is not Shuffle.FALSE:
        if num_shards is not None:
            # If shuffle enabled, sharding enabled, use distributed random sampler
            return DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
        # If shuffle enabled, sharding disabled, use random sampler
        if num_samples is not None:
            return RandomSampler(replacement=True, num_samples=num_samples, shuffle=shuffle)
        return RandomSampler(num_samples=num_samples, shuffle=shuffle)
    if num_shards is not None:
        # If shuffle disabled, sharding enabled, use distributed sequential sampler
        return DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
    # If shuffle disabled, sharding disabled, use sequential sampler
    return SequentialSampler(num_samples=num_samples)


class BuiltinSampler:
    """
    Base class for BuiltinSampler.

    User should not extend this class.
    """

    def __init__(self, num_samples=None):
        self.child_sampler = None
        self.num_samples = num_samples

    def parse(self):
        """ Parse the sampler."""

    def add_child(self, sampler):
        """
        Add a sub-sampler to the given sampler. The parent will receive all data from the
        output of sub-sampler and apply its sample logic to return new samples.

        Note:
            - If a child sampler is added and it has a shuffle option, its value cannot be ``Shuffle.PARTIAL`` .
              Additionally, the parent sampler's shuffle value must be ``Shuffle.GLOBAL`` .

        Args:
            sampler (Sampler): Object used to choose samples from the dataset. Only builtin
                samplers(:class:`mindspore.dataset.DistributedSampler` ,
                :class:`mindspore.dataset.PKSampler`,
                :class:`mindspore.dataset.RandomSampler`,
                :class:`mindspore.dataset.SequentialSampler`,
                :class:`mindspore.dataset.SubsetRandomSampler`,
                :class:`mindspore.dataset.WeightedRandomSampler` ) are supported.

        Examples:
            >>> import mindspore.dataset as ds
            >>> sampler = ds.SequentialSampler(start_index=0, num_samples=3)
            >>> sampler.add_child(ds.RandomSampler(num_samples=4))
            >>> dataset = ds.Cifar10Dataset(cifar10_dataset_dir, sampler=sampler)
        """
        if self.child_sampler is not None:
            raise RuntimeError("Cannot add child sampler, this sampler already has a child.")

        if sampler is not None and sampler.get_shuffle_mode() == Shuffle.PARTIAL:
            raise RuntimeError("When multiple samplers are used, ensure that the shuffle of the input sampler "
                               "must not be Shuffle.PARTIAL.")

        if self.get_shuffle_mode() != Shuffle.GLOBAL and self.get_shuffle_mode() != Shuffle.FALSE:
            raise RuntimeError("When multiple samplers are used, ensure that the shuffle of the current sampler "
                               "must be Shuffle.FALSE or Shuffle.GLOBAL, but got: {}.".format(self.get_shuffle_mode()))

        self.child_sampler = sampler

    def get_child(self):
        """
        Get the child sampler of the given sampler.

        Returns:
            Sampler, the child sampler of the given sampler.

        Examples:
            >>> import mindspore.dataset as ds
            >>> sampler = ds.SequentialSampler(start_index=0, num_samples=3)
            >>> sampler.add_child(ds.RandomSampler(num_samples=2))
            >>> child_sampler = sampler.get_child()
        """
        return self.child_sampler

    def parse_child(self):
        """ Parse the child sampler. """
        c_child_sampler = None
        if self.child_sampler is not None:
            c_child_sampler = self.child_sampler.parse()
        return c_child_sampler

    def parse_child_for_minddataset(self):
        """ Parse the child sampler for MindRecord. """
        c_child_sampler = None
        if self.child_sampler is not None:
            c_child_sampler = self.child_sampler.parse_for_minddataset()
        return c_child_sampler

    def is_shuffled(self):
        """ Not implemented. """
        raise NotImplementedError("Sampler must implement is_shuffled.")

    def is_sharded(self):
        """ Not implemented. """
        raise NotImplementedError("Sampler must implement is_sharded.")

    def get_num_samples(self):
        """
        Get `num_samples` value of the current sampler instance.
        This parameter can be optionally passed in when defining the Sampler. Default: ``None``.
        This method will return the num_samples value.
        If the current sampler has child samplers,
        it will continue to access the child samplers and process the obtained value according to certain rules.

        The following table shows the various possible combinations, and the final results returned.

        .. list-table::
           :widths: 25 25 25 25
           :header-rows: 1

           * - child sampler
             - num_samples
             - child_samples
             - result
           * - T
             - x
             - y
             - min(x, y)
           * - T
             - x
             - None
             - x
           * - T
             - None
             - y
             - y
           * - T
             - None
             - None
             - None
           * - None
             - x
             - n/a
             - x
           * - None
             - None
             - n/a
             - None

        Returns:
            int or None, the number of samples.

        Examples:
            >>> import mindspore.dataset as ds
            >>> sampler = ds.SequentialSampler(start_index=0, num_samples=3)
            >>> num_samplers = sampler.get_num_samples()
        """
        if self.child_sampler is not None:
            child_samples = self.child_sampler.get_num_samples()
            if self.num_samples is not None:
                if child_samples is not None:
                    return min(self.num_samples, child_samples)

                return self.num_samples

            return child_samples

        return self.num_samples

    def get_shuffle_mode(self):
        """ Not implemented. """
        return Shuffle.FALSE


class Sampler(BuiltinSampler):
    """
    Base class for user defined sampler.
    A user defined sampler can be used with any existing dataset with sampler support.

    A required  _iter_() method should by overridden by the user for sample index generation.
    An optional reset() method can be overridden for per repeat reset,

    dataset_size and num_samples will be set by dataset once a dataset iterator is created.

    Examples:
        >>> import mindspore.dataset as ds
        >>> class ReverseSampler(ds.Sampler):
        ...     def __iter__(self):
        ...         for i in range(self.dataset_size - 1, -1, -1):
        ...             yield i
        >>>
        >>> ds = ds.ImageFolderDataset(image_folder_dataset_dir, sampler=ReverseSampler())
    """

    def __init__(self, num_samples=None):
        super().__init__(num_samples)
        self.dataset_size = 0
        self.child_sampler = None
        self.num_samples = num_samples
        if self.num_samples is None and hasattr(self, '__len__'):
            self.num_samples = len(self)
        self.batch_sizes = []

    def __iter__(self):
        """
        User defined iterator, must be overridden.
        _handshake is guaranteed to be called prior to iterator construction.
        """
        raise NotImplementedError

    def reset(self):
        """
        Per repeat reset callback, override this method if necessary
        """

    # Initialization handshake callback
    # Do not override this method!
    def _handshake(self, ds_size):
        self.dataset_size = ds_size

    def get_indices(self):
        """
        Get the indices of the sampler.

        Do not override this method!
        """
        ret = []
        batch_sizes = []
        for count, idx in enumerate(self):
            # The idx can be either a number (for sampler) or a list (for batch sampler).
            # If number, we convert it to list first. So they can be handled in the same way.
            if isinstance(idx, numbers.Number):
                idx = [idx]
                # normal sampler does not have batch sizes
                batch_sizes.append(0)
            else:
                # Using extend instead of append will flatten the list, so we need to save the
                # batch size information here.
                batch_sizes.append(len(idx))
            ret.extend(idx)
            if self.num_samples is not None and count + 1 >= self.num_samples:
                break
        self.batch_sizes.append(batch_sizes)
        indices = np.array(ret)
        if indices.dtype == object:
            raise RuntimeError("Fetched indices can not be converted to a valid ndarray.")
        return indices

    def _get_batch_sizes(self):
        if not self.batch_sizes:
            return []
        return self.batch_sizes.pop(0)

    # Instance fetcher
    # Do not override this method!
    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.PreBuiltSamplerObj(num_samples, self)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def add_child(self, sampler):
        self.child_sampler = sampler

    def get_child(self):
        return self.child_sampler

    def parse_child(self):
        c_child_sampler = None
        if self.child_sampler is not None:
            c_child_sampler = self.child_sampler.parse()

        return c_child_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

    def get_num_samples(self):
        if self.num_samples is not None:
            return self.num_samples
        # deepcopy self to avoid changing the random state
        fake_sampler = copy.deepcopy(self)
        fake_sampler.get_indices()
        return len(fake_sampler.batch_sizes[-1])


class DistributedSampler(BuiltinSampler):
    """
    A sampler that accesses a shard of the dataset, it helps divide dataset into multi-subset for distributed training.

    Note:
        The shuffling modes supported for different datasets are as follows:

        .. list-table:: List of support for shuffling mode
            :widths: 50 50 50 50
            :header-rows: 1

            * - Shuffling Mode
              - MindDataset
              - TFRecordDataset
              - Others
            * - ``Shuffle.ADAPTIVE``
              - Supported
              - Not Supported
              - Not Supported
            * - ``Shuffle.GLOBAL``
              - Supported
              - Supported
              - Supported
            * - ``Shuffle.PARTIAL``
              - Supported
              - Not Supported
              - Not Supported
            * - ``Shuffle.FILES``
              - Supported
              - Supported
              - Not Supported
            * - ``Shuffle.INFILE``
              - Supported
              - Not Supported
              - Not Supported

    Args:
        num_shards (int): Number of shards to divide the dataset into.
        shard_id (int): Shard ID of the current shard, which should be within the range of [0, `num_shards` - 1].
        shuffle (Union[bool, Shuffle], optional): Specify the shuffle mode.
            Default: ``True``, performs ``mindspore.dataset.Shuffle.GLOBAL`` . If `shuffle` is ``False`` ,
            no shuffling will be performed.
            There are several levels of shuffling, desired shuffle enum is
            defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.ADAPTIVE`` : When the number of dataset samples is less than or equal to 100 million,
              ``Shuffle.GLOBAL`` is used. When the number of dataset samples is greater than 100
              million, ``Shuffle.PARTIAL`` is used. The shuffle is performed once every 1 million samples.

            - ``Shuffle.GLOBAL`` : Global shuffle of all rows of data in dataset. The memory usage is large.

            - ``Shuffle.PARTIAL`` : Partial shuffle of data in dataset for every 1 million samples.
              The memory usage is less than ``Shuffle.GLOBAL`` .

            - ``Shuffle.FILES`` : Shuffle the file sequence but keep the order of data within each file.

            - ``Shuffle.INFILE`` : Keep the file sequence the same but shuffle the data within each file.

        num_samples (int, optional): The number of samples to draw. Default: ``None``, which means sample all elements.
        offset (int, optional): The starting shard ID where the elements in the dataset are sent to, which
            should be no more than `num_shards` . This parameter is only valid when a ConcatDataset takes
            a :class:`mindspore.dataset.DistributedSampler` as its sampler. It will affect the number of
            samples per shard. Default: ``-1``, which means each shard has the same number of samples.

    Raises:
        TypeError: If `num_shards` is not of type int.
        TypeError: If `shard_id` is not of type int.
        TypeError: If `shuffle` is not of type bool or Shuffle.
        TypeError: If `num_samples` is not of type int.
        TypeError: If `offset` is not of type int.
        ValueError: If `num_samples` is a negative value.
        RuntimeError: If `num_shards` is not a positive value.
        RuntimeError: If `shard_id` is smaller than 0 or equal to `num_shards` or larger than `num_shards` .
        RuntimeError: If `offset` is greater than `num_shards` .

    Examples:
        >>> import mindspore.dataset as ds
        >>> # creates a distributed sampler with 10 shards in total. This shard is shard 5.
        >>> sampler = ds.DistributedSampler(10, 5)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
    """

    def __init__(self, num_shards, shard_id, shuffle=True, num_samples=None, offset=-1):
        if not isinstance(num_shards, int):
            raise TypeError("num_shards must be integer but was: {}.".format(num_shards))

        if not isinstance(shard_id, int):
            raise TypeError("shard_id must be integer but was: {}.".format(shard_id))

        if not isinstance(shuffle, bool) and shuffle not in (Shuffle.FALSE, Shuffle.ADAPTIVE, Shuffle.GLOBAL,
                                                             Shuffle.PARTIAL, Shuffle.FILES, Shuffle.INFILE):
            raise TypeError("shuffle must be a boolean value or valid shuffle mode but was: {}.".format(shuffle))

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        if not isinstance(offset, int):
            raise TypeError("offset must be integer but was: {}.".format(offset))

        self.num_shards = num_shards
        self.shard_id = shard_id

        if isinstance(shuffle, bool):
            shuffle = Shuffle.GLOBAL if shuffle is True else Shuffle.FALSE
        self.shuffle = shuffle

        # get seed in distributed scenario
        # Example 1. if user set seeds by ds.config.set_seed(4321), then seed 4321 is used
        # Example 2. if user does not set the seed, then existing or default seed (like 5489) is used
        self.seed = ds.config.get_seed()
        self.offset = offset
        super().__init__(num_samples)

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        shuffle = self.shuffle if self.shuffle is not None else True

        if isinstance(shuffle, bool):
            shuffle = Shuffle.GLOBAL if shuffle else Shuffle.FALSE

        if shuffle not in (Shuffle.FALSE, Shuffle.GLOBAL):
            raise RuntimeError("The shuffle mode: {} is not supported with current dataset.".format(self.shuffle))

        offset = self.offset if self.offset is not None else -1
        # each time user calls create_dict_iterator() (to do repeat) sampler would get a different seed to shuffle
        self.seed += 1
        c_sampler = cde.DistributedSamplerObj(self.num_shards, self.shard_id,
                                              shuffle_to_shuffle_mode(shuffle), num_samples, self.seed, offset, True)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def parse_for_minddataset(self):
        """ Parse the sampler for MindRecord."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        shuffle = self.shuffle if self.shuffle is not None else True

        # convert shuffle=True to Shuffle.ADAPTIVE, convert shuffle=False to Shuffle.FALSE
        if isinstance(shuffle, bool):
            if shuffle:
                shuffle = Shuffle.ADAPTIVE
            else:
                shuffle = Shuffle.FALSE
        c_sampler = cde.MindrecordDistributedSampler(self.num_shards, self.shard_id, shuffle_to_shuffle_mode(shuffle),
                                                     self.seed, num_samples, self.offset)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(num_samples)
        return c_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            if self.shuffle == Shuffle.FALSE:
                return False
            return True

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return self.num_shards > 1

        return self.child_sampler.is_sharded()

    def set_offset(self, offset):
        self.offset = offset
        return self

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return self.shuffle


class PKSampler(BuiltinSampler):
    """
    Samples K elements for each P class in the dataset.

    Args:
        num_val (int): Number of elements to sample for each class.
        num_class (int, optional): Number of classes to sample. Default: ``None`` , sample all classes.
            This parameter is not currently supported.
        shuffle (bool, optional): Whether to shuffle the class IDs. Default: ``False``.
        class_column (str, optional): Name of column with class labels for MindDataset. Default: ``'label'``.
        num_samples (int, optional): The number of samples to draw. Default: ``None`` , which means sample all elements.

    Raises:
        TypeError: If `shuffle` is not of type bool.
        TypeError: If `class_column` is not of type str.
        TypeError: If `num_samples` is not of type int.
        NotImplementedError: If `num_class` is not ``None``.
        RuntimeError: If `num_val` is not a positive value.
        ValueError: If `num_samples` is a negative value.

    Examples:
        >>> import mindspore.dataset as ds
        >>> # creates a PKSampler that will get 3 samples from every class.
        >>> sampler = ds.PKSampler(3)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
    """

    def __init__(self, num_val, num_class=None, shuffle=False, class_column='label', num_samples=None):
        if not isinstance(num_val, int):
            raise TypeError("num_val must be integer but was: {}.".format(num_val))

        if num_class is not None:
            raise NotImplementedError("Not supported to specify num_class for PKSampler.")

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean value but was: {}.".format(shuffle))

        if not isinstance(class_column, str):
            raise TypeError("class_column must be a str value but was: {}.".format(class_column))

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        self.num_val = num_val
        self.shuffle = shuffle
        self.class_column = class_column  # work for minddataset
        super().__init__(num_samples)

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        shuffle = self.shuffle if self.shuffle is not None else False
        c_sampler = cde.PKSamplerObj(self.num_val, shuffle, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            return self.shuffle

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

    def parse_for_minddataset(self):
        """Parse the sampler for MindRecord."""
        if not self.class_column or not isinstance(self.class_column, str):
            raise ValueError("class_column should be a not empty string value, \
                    but got class_column: {}.".format(self.class_column))
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordPkSampler(self.num_val, self.class_column, self.shuffle, num_samples)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(num_samples)
        return c_sampler

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return Shuffle.FALSE


class RandomSampler(BuiltinSampler):
    """
    Samples the elements randomly.

    Note:
        The shuffling modes supported for different datasets are as follows:

        .. list-table:: List of support for shuffling mode
            :widths: 50 50 50 50
            :header-rows: 1

            * - Shuffling Mode
              - MindDataset
              - TFRecordDataset
              - Others
            * - ``Shuffle.ADAPTIVE``
              - Supported
              - Not Supported
              - Not Supported
            * - ``Shuffle.GLOBAL``
              - Supported
              - Supported
              - Supported
            * - ``Shuffle.PARTIAL``
              - Supported
              - Not Supported
              - Not Supported
            * - ``Shuffle.FILES``
              - Supported
              - Supported
              - Not Supported
            * - ``Shuffle.INFILE``
              - Supported
              - Not Supported
              - Not Supported

    Args:
        replacement (bool, optional): If True, put the sample ID back for the next draw. Default: ``False``.
        num_samples (int, optional): Number of elements to sample. Default: ``None`` , which means sample all elements.
        shuffle (Shuffle, optional): Specify the shuffle mode.
            Default: ``Shuffle.GLOBAL``, Global shuffle of all rows of data in dataset.
            There are several levels of shuffling, desired shuffle enum defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.ADAPTIVE`` : When the number of dataset samples is less than or equal to 100 million,
              ``Shuffle.GLOBAL`` is used. When the number of dataset samples is greater than 100
              million, ``Shuffle.PARTIAL`` is used. The shuffle is performed once every 1 million samples.

            - ``Shuffle.GLOBAL`` : Global shuffle of all rows of data in dataset. The memory usage is large.

            - ``Shuffle.PARTIAL`` : Partial shuffle of data in dataset for every 1 million samples.
              The memory usage is less than ``Shuffle.GLOBAL`` .

            - ``Shuffle.FILES`` : Shuffle the file sequence but keep the order of data within each file.

            - ``Shuffle.INFILE`` : Keep the file sequence the same but shuffle the data within each file.

    Raises:
        TypeError: If `replacement` is not of type bool.
        TypeError: If `num_samples` is not of type int.
        ValueError: If `num_samples` is a negative value.
        TypeError: If `shuffle` is not of type Shuffle.

    Examples:
        >>> import mindspore.dataset as ds
        >>> # creates a RandomSampler
        >>> sampler = ds.RandomSampler()
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
     """

    def __init__(self, replacement=False, num_samples=None, shuffle=Shuffle.GLOBAL):
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be a boolean value but was: {}.".format(replacement))

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        if shuffle not in (Shuffle.ADAPTIVE, Shuffle.GLOBAL, Shuffle.PARTIAL, Shuffle.FILES, Shuffle.INFILE):
            raise TypeError("shuffle must be valid shuffle mode but was: {}.".format(shuffle))
        self.shuffle = shuffle

        self.deterministic = False
        self.replacement = replacement
        self.reshuffle_each_epoch = True
        super().__init__(num_samples)

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        replacement = self.replacement if self.replacement is not None else False
        # convert shuffle=True to Shuffle.GLOBAL, convert shuffle=False to Shuffle.FALSE
        if self.shuffle is not Shuffle.GLOBAL:
            raise RuntimeError("The shuffle mode: {} is not supported with current dataset.".format(self.shuffle))
        c_sampler = cde.RandomSamplerObj(replacement, num_samples, self.reshuffle_each_epoch,
                                         shuffle_to_shuffle_mode(self.shuffle))
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def parse_for_minddataset(self):
        """Parse the sampler for MindRecord."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        shuffle = self.shuffle if self.shuffle is not None else True
        # convert shuffle=True to Shuffle.ADAPTIVE, convert shuffle=False to Shuffle.FALSE
        if isinstance(shuffle, bool):
            if shuffle:
                shuffle = Shuffle.ADAPTIVE
            else:
                raise RuntimeError("The shuffle: False is invalid for RandomSampler.")
        c_sampler = cde.MindrecordRandomSampler(num_samples, self.replacement, self.reshuffle_each_epoch,
                                                shuffle_to_shuffle_mode(shuffle))
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(num_samples)
        return c_sampler

    def is_shuffled(self):
        return True

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return self.shuffle


class SequentialSampler(BuiltinSampler):
    """
    Samples the dataset elements sequentially that is equivalent to not using a sampler.

    Args:
        start_index (int, optional): Index to start sampling at. Default: ``None`` , start at first ID.
        num_samples (int, optional): Number of elements to sample. Default: ``None`` , which means sample all elements.

    Raises:
        TypeError: If `start_index` is not of type int.
        TypeError: If `num_samples` is not of type int.
        RuntimeError: If `start_index` is a negative value.
        ValueError: If `num_samples` is a negative value.

    Examples:
        >>> import mindspore.dataset as ds
        >>> # creates a SequentialSampler
        >>> sampler = ds.SequentialSampler()
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
    """

    def __init__(self, start_index=None, num_samples=None):
        if start_index is not None and not isinstance(start_index, int):
            raise TypeError("start_index must be integer but was: {}.".format(start_index))

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        self.start_index = start_index
        super().__init__(num_samples)

    def parse(self):
        """ Parse the sampler."""
        start_index = self.start_index if self.start_index is not None else 0
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SequentialSamplerObj(start_index, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def parse_for_minddataset(self):
        """Parse the sampler for MindRecord."""
        start_index = self.start_index if self.start_index is not None else 0
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordSequentialSampler(num_samples, start_index)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(num_samples)
        return c_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return Shuffle.FALSE


class SubsetSampler(BuiltinSampler):
    """
    Samples the elements from a sequence of indices.

    Args:
        indices (Iterable): A sequence of indices (Any iterable Python object but string).
        num_samples (int, optional): Number of elements to sample.
            Used to partially extract samples obtained through sampling.
            Default: ``None`` , which means sample all elements.

    Raises:
        TypeError: If elements of `indices` are not of type number.
        TypeError: If `num_samples` is not of type int.
        ValueError: If `num_samples` is a negative value.

    Examples:
        >>> import mindspore.dataset as ds
        >>> indices = [0, 1, 2, 3, 4, 5]
        >>>
        >>> # creates a SubsetSampler, will sample from the provided indices
        >>> sampler = ds.SubsetSampler(indices)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
    """

    def __init__(self, indices, num_samples=None):
        def _get_sample_ids_as_list(sampler, number_of_samples=None):
            if number_of_samples is None:
                return list(sampler)

            if isinstance(sampler, list):
                return sampler[:number_of_samples]

            return [sample_id for sample_id, _ in zip(sampler, range(number_of_samples))]

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        if not isinstance(indices, str) and validator.is_iterable(indices):
            indices = _get_sample_ids_as_list(indices, num_samples)
        elif isinstance(indices, int):
            indices = [indices]
        else:
            raise TypeError('Unsupported sampler object of type ({})'.format(type(indices)))

        for i, item in enumerate(indices):
            if not isinstance(item, (int, np.integer)):
                raise TypeError("SubsetSampler: Type of indices element must be int, "
                                "but got list[{}]: {}, type: {}.".format(i, item, type(item)))

        self.indices = indices
        super().__init__(num_samples)

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SubsetSamplerObj(self.indices, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        return False

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

    def parse_for_minddataset(self):
        """Parse the sampler for MindRecord."""
        c_sampler = cde.MindrecordSubsetSampler(self.indices)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(self.get_num_samples())
        return c_sampler

    def get_num_samples(self):
        num_samples = super().get_num_samples()
        if num_samples is None:
            return len(self.indices)

        return min(len(self.indices), num_samples)

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return Shuffle.FALSE


class SubsetRandomSampler(SubsetSampler):
    """
    Samples the elements randomly from a sequence of indices.

    Args:
        indices (Iterable): A sequence of indices (Any iterable Python object but string).
        num_samples (int, optional): Number of elements to sample.
            Used to partially extract samples obtained through sampling.
            Default: ``None`` , which means sample all elements.

    Raises:
        TypeError: If elements of `indices` are not of type number.
        TypeError: If `num_samples` is not of type int.
        ValueError: If `num_samples` is a negative value.

    Examples:
        >>> import mindspore.dataset as ds
        >>> indices = [0, 1, 2, 3, 7, 88, 119]
        >>>
        >>> # create a SubsetRandomSampler, will sample from the provided indices
        >>> sampler = ds.SubsetRandomSampler(indices)
        >>> data = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8, sampler=sampler)
    """

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SubsetRandomSamplerObj(self.indices, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        return True

    def parse_for_minddataset(self):
        """Parse the sampler for MindRecord."""
        c_sampler = cde.MindrecordSubsetSampler(self.indices, ds.config.get_seed())
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        c_sampler.set_num_samples(self.get_num_samples())
        return c_sampler

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return Shuffle.GLOBAL


class IterSampler(Sampler):
    """
    User provided an iterable object without inheriting from our Sampler class.

    Note:
        This class exists to allow handshake logic between dataset operations and user defined samplers.
        By constructing this object we avoid the user having to inherit from our Sampler class.

    Args:
        sampler (iterable object): an user defined iterable object.
        num_samples (int, optional): Number of elements to sample. Default: ``None`` , which means sample all elements.

    Examples:
        >>> import mindspore.dataset as ds
        >>> class MySampler:
        ...     def __iter__(self):
        ...         for i in range(99, -1, -1):
        ...             yield i

        >>> # creates an IterSampler
        >>> sampler = ds.IterSampler(sampler=MySampler())
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
     """

    def __init__(self, sampler, num_samples=None):
        if num_samples is None and hasattr(sampler, '__len__'):
            num_samples = len(sampler)
        super().__init__(num_samples=num_samples)
        self.sampler = sampler

    def __iter__(self):
        return iter(self.sampler)


class WeightedRandomSampler(BuiltinSampler):
    """
    Samples the elements from [0, len(weights) - 1] randomly with the given weights (probabilities).

    Args:
        weights (list[float, int]): A sequence of weights, not necessarily summing up to 1.
        num_samples (int, optional): Number of elements to sample. Default: ``None`` ,
            which means sample all elements.
        replacement (bool, optional): If ``True``, put the sample ID back for the next draw. Default: ``True``.

    Raises:
        TypeError: If elements of `weights` are not of type number.
        TypeError: If `num_samples` is not of type int.
        TypeError: If `replacement` is not of type bool.
        RuntimeError: If `weights` is empty or all zero.
        ValueError: If `num_samples` is a negative value.

    Examples:
        >>> import mindspore.dataset as ds
        >>> weights = [0.9, 0.01, 0.4, 0.8, 0.1, 0.1, 0.3]
        >>>
        >>> # creates a WeightedRandomSampler that will sample 4 elements without replacement
        >>> sampler = ds.WeightedRandomSampler(weights, 4)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)
    """

    def __init__(self, weights, num_samples=None, replacement=True):
        if not isinstance(weights, list):
            weights = [weights]

        for ind, w in enumerate(weights):
            if not isinstance(w, numbers.Number):
                raise TypeError("type of weights element must be number, "
                                "but got w[{}]: {}, type: {}.".format(ind, w, type(w)))

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        if not isinstance(replacement, bool):
            raise TypeError("replacement must be a boolean value but was: {}.".format(replacement))

        self.weights = weights
        self.replacement = replacement
        super().__init__(num_samples)

    def parse(self):
        """ Parse the sampler."""
        num_samples = self.num_samples if self.num_samples is not None else 0
        replacement = self.replacement if self.replacement is not None else True
        c_sampler = cde.WeightedRandomSamplerObj(self.weights, num_samples, replacement)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        return True

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

    def get_shuffle_mode(self):
        """Get the shuffle mode"""
        return Shuffle.GLOBAL
