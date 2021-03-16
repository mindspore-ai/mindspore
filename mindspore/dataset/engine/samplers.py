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

import numbers
import numpy as np
import mindspore._c_dataengine as cde
import mindspore.dataset as ds
from ..core import validator_helpers as validator


def select_sampler(num_samples, input_sampler, shuffle, num_shards, shard_id):
    """
    Create sampler based on user input.

    Args:
        num_samples (int): Number of samples.
        input_sampler (Union[Iterable, Sampler]): Sampler from user.
        shuffle (bool): Shuffle.
        num_shards (int): Number of shard for sharding.
        shard_id (int): Shard ID.

    Returns:
        Sampler, sampler selected based on user input.
    """

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
        if not isinstance(input_sampler, str) and isinstance(input_sampler, (np.ndarray, list)):
            return SubsetSampler(input_sampler, num_samples)
        if not isinstance(input_sampler, str) and validator.is_iterable(input_sampler):
            # in this case, the user passed in their own sampler object that's not of type BuiltinSampler
            return IterSampler(input_sampler, num_samples)
        if isinstance(input_sampler, int):
            return SubsetSampler([input_sampler])
        raise TypeError('Unsupported sampler object of type ({})'.format(type(input_sampler)))
    if shuffle is None:
        if num_shards is not None:
            # If shuffle is not specified, sharding enabled, use distributed random sampler
            shuffle = True
            return DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
        # If shuffle is not specified, sharding disabled, use random sampler
        if num_samples is not None and num_samples != 0:
            return RandomSampler(replacement=True, num_samples=num_samples)
        return RandomSampler(num_samples=num_samples)
    if shuffle is True:
        if num_shards is not None:
            # If shuffle enabled, sharding enabled, use distributed random sampler
            return DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
        # If shuffle enabled, sharding disabled, use random sampler
        if num_samples is not None:
            return RandomSampler(replacement=True, num_samples=num_samples)
        return RandomSampler(num_samples=num_samples)
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
        pass

    def add_child(self, sampler):
        """
        Add a sub-sampler for given sampler. The sub-sampler will receive all data from the
        output of parent sampler and apply its sample logic to return new samples.

        Args:
            sampler (Sampler): Object used to choose samples from the dataset. Only builtin
                samplers(DistributedSampler, PKSampler, RandomSampler, SequentialSampler,
                SubsetRandomSampler, WeightedRandomSampler) are supported.

        Examples:
            >>> sampler = ds.SequentialSampler(start_index=0, num_samples=3)
            >>> sampler.add_child(ds.RandomSampler(num_samples=2))
            >>> dataset = ds.Cifar10Dataset(cifar10_dataset_dir, sampler=sampler)
        """
        self.child_sampler = sampler

    def get_child(self):
        return self.child_sampler

    def parse_child(self):
        c_child_sampler = None
        if self.child_sampler is not None:
            c_child_sampler = self.child_sampler.parse()
        return c_child_sampler

    def parse_child_for_minddataset(self):
        c_child_sampler = None
        if self.child_sampler is not None:
            c_child_sampler = self.child_sampler.parse_for_minddataset()
        return c_child_sampler

    def is_shuffled(self):
        raise NotImplementedError("Sampler must implement is_shuffled.")

    def is_sharded(self):
        raise NotImplementedError("Sampler must implement is_sharded.")

    def get_num_samples(self):
        """
        All samplers can contain a numeric num_samples value (or it can be set to None).
        A child sampler can exist or be None.
        If a child sampler exists, then the child sampler count can be a numeric value or None.
        These conditions impact the resultant sampler count that is used.
        The following table shows the possible results from calling this function.

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
            int, the number of samples, or None
        """
        if self.child_sampler is not None:
            child_samples = self.child_sampler.get_num_samples()
            if self.num_samples is not None:
                if child_samples is not None:
                    return min(self.num_samples, child_samples)

                return self.num_samples

            return child_samples

        return self.num_samples


class Sampler(BuiltinSampler):
    """
    Base class for user defined sampler.
    A user defined sampler can be used with any existing dataset with sampler support.

    A required  _iter_() method should by overridden by the user for sample index generation.
    An optional reset() method can be overridden for per repeat reset,

    dataset_size and num_samples will be set by dataset once a dataset iterator is created.

    Examples:
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
    def _handshake(self, ds_size, num_samples):
        self.dataset_size = ds_size
        self.num_samples = num_samples

    # Indices fetcher
    # Do not override this method!
    # pylint: disable=missing-docstring
    def _get_indices(self):
        sampler_iter = iter(self)
        ret = []
        for _ in range(self.num_samples):
            try:
                idx = next(sampler_iter)
                ret.append(idx)
            except StopIteration:
                break
        indices = np.array(ret)
        if indices.dtype == object:
            raise RuntimeError("Fetched indices can not be converted to a valid ndarray.")
        return indices

    # Instance fetcher
    # Do not override this method!
    def parse(self):
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
        if self.num_samples is None:
            return None
        return self._get_indices().size


class DistributedSampler(BuiltinSampler):
    """
    A sampler that accesses a shard of the dataset.

    Args:
        num_shards (int): Number of shards to divide the dataset into.
        shard_id (int): Shard ID of the current shard within num_shards.
        shuffle (bool, optional): If True, the indices are shuffled (default=True).
        num_samples (int, optional): The number of samples to draw (default=None, all elements).
        offset(int, optional): The starting shard ID where the elements in the dataset are sent to (default=-1), which
            should be no more than num_shards.

    Examples:
        >>> # creates a distributed sampler with 10 shards in total. This shard is shard 5.
        >>> sampler = ds.DistributedSampler(10, 5)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    Raises:
        TypeError: If num_shards is not an integer value.
        TypeError: If shard_id is not an integer value.
        TypeError: If shuffle is not a boolean value.
        TypeError: If num_samples is not an integer value.
        TypeError: If offset is not an integer value.
        RuntimeError: If num_shards is not a positive value.
        RuntimeError: If shard_id is smaller than 0 or equal to num_shards or larger than num_shards.
        RuntimeError: If num_samples is a negative value.
        RuntimeError: If offset is greater than num_shards.
    """

    def __init__(self, num_shards, shard_id, shuffle=True, num_samples=None, offset=-1):
        if not isinstance(num_shards, int):
            raise TypeError("num_shards must be integer but was: {}.".format(num_shards))

        if not isinstance(shard_id, int):
            raise TypeError("shard_id must be integer but was: {}.".format(shard_id))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean value but was: {}.".format(shuffle))

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
        self.shuffle = shuffle
        self.seed = 0
        self.offset = offset
        super().__init__(num_samples)

    def parse(self):
        num_samples = self.num_samples if self.num_samples is not None else 0
        shuffle = self.shuffle if self.shuffle is not None else True
        offset = self.offset if self.offset is not None else -1
        # each time user calls create_dict_iterator() (to do repeat) sampler would get a different seed to shuffle
        self.seed += 1
        c_sampler = cde.DistributedSamplerObj(self.num_shards, self.shard_id,
                                              shuffle, num_samples, self.seed, offset, True)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def parse_for_minddataset(self):
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordDistributedSampler(self.num_shards, self.shard_id, self.shuffle,
                                                     self.seed, num_samples, self.offset)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            return self.shuffle

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return self.num_shards > 1

        return self.child_sampler.is_sharded()

    def set_offset(self, offset):
        self.offset = offset
        return self


class PKSampler(BuiltinSampler):
    """
    Samples K elements for each P class in the dataset.

    Args:
        num_val (int): Number of elements to sample for each class.
        num_class (int, optional): Number of classes to sample (default=None, all classes).
            The parameter does not supported to specify currently.
        shuffle (bool, optional): If True, the class IDs are shuffled (default=False).
        class_column (str, optional): Name of column with class labels for MindDataset (default='label').
        num_samples (int, optional): The number of samples to draw (default=None, all elements).

    Examples:
        >>> # creates a PKSampler that will get 3 samples from every class.
        >>> sampler = ds.PKSampler(3)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    Raises:
        TypeError: If num_val is not a positive value.
        TypeError: If shuffle is not a boolean value.
        TypeError: If class_column is not a str value.
        TypeError: If num_samples is not an integer value.
        NotImplementedError: If num_class is not None.
        RuntimeError: If num_val is not a positive value.
        RuntimeError: If num_samples is a negative value.
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
        if not self.class_column or not isinstance(self.class_column, str):
            raise ValueError("class_column should be a not empty string value, \
                    but got class_column: {}.".format(class_column))
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordPkSampler(self.num_val, self.class_column, self.shuffle, num_samples)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler


class RandomSampler(BuiltinSampler):
    """
    Samples the elements randomly.

    Args:
        replacement (bool, optional): If True, put the sample ID back for the next draw (default=False).
        num_samples (int, optional): Number of elements to sample (default=None, all elements).

    Examples:
        >>> # creates a RandomSampler
        >>> sampler = ds.RandomSampler()
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    Raises:
        TypeError: If replacement is not a boolean value.
        TypeError: If num_samples is not an integer value.
        RuntimeError: If num_samples is a negative value.
     """

    def __init__(self, replacement=False, num_samples=None):
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be a boolean value but was: {}.".format(replacement))

        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise TypeError("num_samples must be integer but was: {}.".format(num_samples))
            if num_samples < 0 or num_samples > validator.INT64_MAX:
                raise ValueError("num_samples exceeds the boundary between {} and {}(INT64_MAX)!"
                                 .format(0, validator.INT64_MAX))

        self.deterministic = False
        self.replacement = replacement
        self.reshuffle_each_epoch = True
        super().__init__(num_samples)

    def parse(self):
        num_samples = self.num_samples if self.num_samples is not None else 0
        replacement = self.replacement if self.replacement is not None else False
        c_sampler = cde.RandomSamplerObj(replacement, num_samples, self.reshuffle_each_epoch)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def parse_for_minddataset(self):
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordRandomSampler(num_samples, self.replacement, self.reshuffle_each_epoch)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        return True

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()


class SequentialSampler(BuiltinSampler):
    """
    Samples the dataset elements sequentially, same as not having a sampler.

    Args:
        start_index (int, optional): Index to start sampling at. (default=None, start at first ID)
        num_samples (int, optional): Number of elements to sample (default=None, all elements).

    Examples:
        >>> # creates a SequentialSampler
        >>> sampler = ds.SequentialSampler()
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    Raises:
        TypeError: If start_index is not an integer value.
        TypeError: If num_samples is not an integer value.
        RuntimeError: If start_index is a negative value.
        RuntimeError: If num_samples is a negative value.
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
        start_index = self.start_index if self.start_index is not None else 0
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SequentialSamplerObj(start_index, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def parse_for_minddataset(self):
        start_index = self.start_index if self.start_index is not None else 0
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordSequentialSampler(num_samples, start_index)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()


class SubsetSampler(BuiltinSampler):
    """
    Samples the elements from a sequence of indices.

    Args:
        indices (Any iterable python object but string): A sequence of indices.
        num_samples (int, optional): Number of elements to sample (default=None, all elements).

    Examples:
        >>> indices = [0, 1, 2, 3, 4, 5]
        >>>
        >>> # creates a SubsetSampler, will sample from the provided indices
        >>> sampler = ds.SubsetSampler(indices)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    Raises:
        TypeError: If type of indices element is not a number.
        TypeError: If num_samples is not an integer value.
        RuntimeError: If num_samples is a negative value.
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
        c_sampler = cde.MindrecordSubsetSampler(self.indices)
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def get_num_samples(self):
        num_samples = super().get_num_samples()
        if num_samples is None:
            return len(self.indices)

        return min(len(self.indices), num_samples)


class SubsetRandomSampler(SubsetSampler):
    """
    Samples the elements randomly from a sequence of indices.

    Args:
        indices (Any iterable python object but string): A sequence of indices.
        num_samples (int, optional): Number of elements to sample (default=None, all elements).

    Examples:
        >>> indices = [0, 1, 2, 3, 7, 88, 119]
        >>>
        >>> # create a SubsetRandomSampler, will sample from the provided indices
        >>> sampler = ds.SubsetRandomSampler(indices)
        >>> data = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8, sampler=sampler)

    Raises:
        TypeError: If type of indices element is not a number.
        TypeError: If num_samples is not an integer value.
        RuntimeError: If num_samples is a negative value.
    """

    def parse(self):
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SubsetRandomSamplerObj(self.indices, num_samples)
        c_child_sampler = self.parse_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        return True

    def parse_for_minddataset(self):
        c_sampler = cde.MindrecordSubsetSampler(self.indices, ds.config.get_seed())
        c_child_sampler = self.parse_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler


class IterSampler(Sampler):
    """
    User provided an iterable object without inheriting from our Sampler class.

    Note:
        This class exists to allow handshake logic between dataset operators and user defined samplers.
        By constructing this object we avoid the user having to inherit from our Sampler class.

    Args:
        sampler (iterable object): an user defined iterable object.
        num_samples (int, optional): Number of elements to sample (default=None, all elements).

    Examples:
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
        if num_samples is None:
            num_samples = len(list(sampler))
        super().__init__(num_samples=num_samples)
        self.sampler = sampler

    def __iter__(self):
        return iter(self.sampler)


class WeightedRandomSampler(BuiltinSampler):
    """
    Samples the elements from [0, len(weights) - 1] randomly with the given weights (probabilities).

    Args:
        weights (list[float, int]): A sequence of weights, not necessarily summing up to 1.
        num_samples (int, optional): Number of elements to sample (default=None, all elements).
        replacement (bool): If True, put the sample ID back for the next draw (default=True).

    Examples:
        >>> weights = [0.9, 0.01, 0.4, 0.8, 0.1, 0.1, 0.3]
        >>>
        >>> # creates a WeightedRandomSampler that will sample 4 elements without replacement
        >>> sampler = ds.WeightedRandomSampler(weights, 4)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    Raises:
        TypeError: If type of weights element is not a number.
        TypeError: If num_samples is not an integer value.
        TypeError: If replacement is not a boolean value.
        RuntimeError: If weights is empty or all zero.
        RuntimeError: If num_samples is a negative value.
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
