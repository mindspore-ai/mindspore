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
# ============================================================================
"""Dataset loading, creation and processing"""
import logging
import math
import os
import time
import timeit
import pickle
import numpy as np
import pandas as pd

from mindspore.dataset import GeneratorDataset, Sampler

import src.constants as rconst
import src.movielens as movielens
import src.stat_utils as stat_utils

DATASET_TO_NUM_USERS_AND_ITEMS = {
    "ml-1m": (6040, 3706),
    "ml-20m": (138493, 26744)
}

_EXPECTED_CACHE_KEYS = (
    rconst.TRAIN_USER_KEY, rconst.TRAIN_ITEM_KEY, rconst.EVAL_USER_KEY,
    rconst.EVAL_ITEM_KEY, rconst.USER_MAP, rconst.ITEM_MAP)


def load_data(data_dir, dataset):
    """
    Load data in .csv format and output structured data.

    This function reads in the raw CSV of positive items, and performs three
    preprocessing transformations:

    1)  Filter out all users who have not rated at least a certain number
      of items. (Typically 20 items)

    2)  Zero index the users and items such that the largest user_id is
      `num_users - 1` and the largest item_id is `num_items - 1`

    3)  Sort the dataframe by user_id, with timestamp as a secondary sort key.
      This allows the dataframe to be sliced by user in-place, and for the last
      item to be selected simply by calling the `-1` index of a user's slice.

    While all of these transformations are performed by Pandas (and are therefore
    single-threaded), they only take ~2 minutes, and the overhead to apply a
    MapReduce pattern to parallel process the dataset adds significant complexity
    for no computational gain. For a larger dataset parallelizing this
    preprocessing could yield speedups. (Also, this preprocessing step is only
    performed once for an entire run.
    """
    logging.info("Beginning loading data...")

    raw_rating_path = os.path.join(data_dir, dataset, movielens.RATINGS_FILE)
    cache_path = os.path.join(data_dir, dataset, rconst.RAW_CACHE_FILE)

    valid_cache = os.path.exists(cache_path)
    if valid_cache:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)

        for key in _EXPECTED_CACHE_KEYS:
            if key not in cached_data:
                valid_cache = False

        if not valid_cache:
            logging.info("Removing stale raw data cache file.")
            os.remove(cache_path)

    if valid_cache:
        data = cached_data
    else:
        # process data and save to .csv
        with open(raw_rating_path) as f:
            df = pd.read_csv(f)

        # Get the info of users who have more than 20 ratings on items
        grouped = df.groupby(movielens.USER_COLUMN)
        df = grouped.filter(lambda x: len(x) >= rconst.MIN_NUM_RATINGS)

        original_users = df[movielens.USER_COLUMN].unique()
        original_items = df[movielens.ITEM_COLUMN].unique()

        # Map the ids of user and item to 0 based index for following processing
        logging.info("Generating user_map and item_map...")
        user_map = {user: index for index, user in enumerate(original_users)}
        item_map = {item: index for index, item in enumerate(original_items)}

        df[movielens.USER_COLUMN] = df[movielens.USER_COLUMN].apply(
            lambda user: user_map[user])
        df[movielens.ITEM_COLUMN] = df[movielens.ITEM_COLUMN].apply(
            lambda item: item_map[item])

        num_users = len(original_users)
        num_items = len(original_items)

        assert num_users <= np.iinfo(rconst.USER_DTYPE).max
        assert num_items <= np.iinfo(rconst.ITEM_DTYPE).max
        assert df[movielens.USER_COLUMN].max() == num_users - 1
        assert df[movielens.ITEM_COLUMN].max() == num_items - 1

        # This sort is used to shard the dataframe by user, and later to select
        # the last item for a user to be used in validation.
        logging.info("Sorting by user, timestamp...")

        # This sort is equivalent to
        #   df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
        #   inplace=True)
        # except that the order of items with the same user and timestamp are
        # sometimes different. For some reason, this sort results in a better
        # hit-rate during evaluation, matching the performance of the MLPerf
        # reference implementation.
        df.sort_values(by=movielens.TIMESTAMP_COLUMN, inplace=True)
        df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                       inplace=True, kind="mergesort")

        # The dataframe does not reconstruct indices in the sort or filter steps.
        df = df.reset_index()

        grouped = df.groupby(movielens.USER_COLUMN, group_keys=False)
        eval_df, train_df = grouped.tail(1), grouped.apply(lambda x: x.iloc[:-1])

        data = {
            rconst.TRAIN_USER_KEY:
                train_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE),
            rconst.TRAIN_ITEM_KEY:
                train_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE),
            rconst.EVAL_USER_KEY:
                eval_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE),
            rconst.EVAL_ITEM_KEY:
                eval_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE),
            rconst.USER_MAP: user_map,
            rconst.ITEM_MAP: item_map,
            "create_time": time.time(),
        }

        logging.info("Writing raw data cache.")
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    num_users, num_items = DATASET_TO_NUM_USERS_AND_ITEMS[dataset]
    if num_users != len(data[rconst.USER_MAP]):
        raise ValueError("Expected to find {} users, but found {}".format(
            num_users, len(data[rconst.USER_MAP])))
    if num_items != len(data[rconst.ITEM_MAP]):
        raise ValueError("Expected to find {} items, but found {}".format(
            num_items, len(data[rconst.ITEM_MAP])))

    return data, num_users, num_items


def construct_lookup_variables(train_pos_users, train_pos_items, num_users):
    """Lookup variables"""
    index_bounds = None
    sorted_train_pos_items = None

    def index_segment(user):
        lower, upper = index_bounds[user:user + 2]
        items = sorted_train_pos_items[lower:upper]

        negatives_since_last_positive = np.concatenate(
            [items[0][np.newaxis], items[1:] - items[:-1] - 1])

        return np.cumsum(negatives_since_last_positive)

    start_time = timeit.default_timer()
    inner_bounds = np.argwhere(train_pos_users[1:] -
                               train_pos_users[:-1])[:, 0] + 1
    (upper_bound,) = train_pos_users.shape
    index_bounds = np.array([0] + inner_bounds.tolist() + [upper_bound])

    # Later logic will assume that the users are in sequential ascending order.
    assert np.array_equal(train_pos_users[index_bounds[:-1]], np.arange(num_users))

    sorted_train_pos_items = train_pos_items.copy()

    for i in range(num_users):
        lower, upper = index_bounds[i:i + 2]
        sorted_train_pos_items[lower:upper].sort()

    total_negatives = np.concatenate([
        index_segment(i) for i in range(num_users)])

    logging.info("Negative total vector built. Time: {:.1f} seconds".format(
        timeit.default_timer() - start_time))

    return total_negatives, index_bounds, sorted_train_pos_items


class NCFDataset:
    """
    A dataset for NCF network.
    """

    def __init__(self,
                 pos_users,
                 pos_items,
                 num_users,
                 num_items,
                 batch_size,
                 total_negatives,
                 index_bounds,
                 sorted_train_pos_items,
                 num_neg,
                 is_training=True):
        self._pos_users = pos_users
        self._pos_items = pos_items
        self._num_users = num_users
        self._num_items = num_items

        self._batch_size = batch_size

        self._total_negatives = total_negatives
        self._index_bounds = index_bounds
        self._sorted_train_pos_items = sorted_train_pos_items

        self._is_training = is_training

        if self._is_training:
            self._train_pos_count = self._pos_users.shape[0]
        else:
            self._eval_users_per_batch = int(
                batch_size // (1 + rconst.NUM_EVAL_NEGATIVES))

        _pos_count = pos_users.shape[0]
        _num_samples = (1 + num_neg) * _pos_count
        self.dataset_len = math.ceil(_num_samples / batch_size)

    def lookup_negative_items(self, negative_users):
        """Lookup negative items"""
        output = np.zeros(shape=negative_users.shape, dtype=rconst.ITEM_DTYPE) - 1

        left_index = self._index_bounds[negative_users]
        right_index = self._index_bounds[negative_users + 1] - 1

        num_positives = right_index - left_index + 1
        num_negatives = self._num_items - num_positives
        neg_item_choice = stat_utils.very_slightly_biased_randint(num_negatives)

        # Shortcuts:
        # For points where the negative is greater than or equal to the tally before
        # the last positive point there is no need to bisect. Instead the item id
        # corresponding to the negative item choice is simply:
        #   last_postive_index + 1 + (neg_choice - last_negative_tally)
        # Similarly, if the selection is less than the tally at the first positive
        # then the item_id is simply the selection.
        #
        # Because MovieLens organizes popular movies into low integers (which is
        # preserved through the preprocessing), the first shortcut is very
        # efficient, allowing ~60% of samples to bypass the bisection. For the same
        # reason, the second shortcut is rarely triggered (<0.02%) and is therefore
        # not worth implementing.
        use_shortcut = neg_item_choice >= self._total_negatives[right_index]
        output[use_shortcut] = (
            self._sorted_train_pos_items[right_index] + 1 +
            (neg_item_choice - self._total_negatives[right_index])
            )[use_shortcut]

        if np.all(use_shortcut):
            # The bisection code is ill-posed when there are no elements.
            return output

        not_use_shortcut = np.logical_not(use_shortcut)
        left_index = left_index[not_use_shortcut]
        right_index = right_index[not_use_shortcut]
        neg_item_choice = neg_item_choice[not_use_shortcut]

        num_loops = np.max(
            np.ceil(np.log2(num_positives[not_use_shortcut])).astype(np.int32))

        for _ in range(num_loops):
            mid_index = (left_index + right_index) // 2
            right_criteria = self._total_negatives[mid_index] > neg_item_choice
            left_criteria = np.logical_not(right_criteria)

            right_index[right_criteria] = mid_index[right_criteria]
            left_index[left_criteria] = mid_index[left_criteria]

        # Expected state after bisection pass:
        #   The right index is the smallest index whose tally is greater than the
        #   negative item choice index.

        assert np.all((right_index - left_index) <= 1)

        output[not_use_shortcut] = (
            self._sorted_train_pos_items[right_index] - (self._total_negatives[right_index] - neg_item_choice)
            )

        assert np.all(output >= 0)

        return output

    def _get_train_item(self, index):
        """Get train item"""
        (mask_start_index,) = index.shape
        index_mod = np.mod(index, self._train_pos_count)

        # get batch of users
        users = self._pos_users[index_mod]

        # get batch of items
        negative_indices = np.greater_equal(index, self._train_pos_count)
        negative_users = users[negative_indices]
        negative_items = self.lookup_negative_items(negative_users=negative_users)
        items = self._pos_items[index_mod]
        items[negative_indices] = negative_items

        # get batch of labels
        labels = np.logical_not(negative_indices)

        # pad last partial batch
        pad_length = self._batch_size - index.shape[0]
        if pad_length:
            user_pad = np.arange(pad_length, dtype=users.dtype) % self._num_users
            item_pad = np.arange(pad_length, dtype=items.dtype) % self._num_items
            label_pad = np.zeros(shape=(pad_length,), dtype=labels.dtype)
            users = np.concatenate([users, user_pad])
            items = np.concatenate([items, item_pad])
            labels = np.concatenate([labels, label_pad])

        users = np.reshape(users, (self._batch_size, 1))  # (_batch_size, 1), int32
        items = np.reshape(items, (self._batch_size, 1))  # (_batch_size, 1), int32
        mask_start_index = np.array(mask_start_index, dtype=np.int32)  # (_batch_size, 1), int32
        valid_pt_mask = np.expand_dims(
            np.less(np.arange(self._batch_size), mask_start_index), -1).astype(np.float32)  # (_batch_size, 1), bool
        labels = np.reshape(labels, (self._batch_size, 1)).astype(np.int32)  # (_batch_size, 1), bool

        return users, items, labels, valid_pt_mask

    @staticmethod
    def _assemble_eval_batch(users, positive_items, negative_items,
                             users_per_batch):
        """Construct duplicate_mask and structure data accordingly.

        The positive items should be last so that they lose ties. However, they
        should not be masked out if the true eval positive happens to be
        selected as a negative. So instead, the positive is placed in the first
        position, and then switched with the last element after the duplicate
        mask has been computed.

        Args:
          users: An array of users in a batch. (should be identical along axis 1)
          positive_items: An array (batch_size x 1) of positive item indices.
          negative_items: An array of negative item indices.
          users_per_batch: How many users should be in the batch. This is passed
            as an argument so that ncf_test.py can use this method.

        Returns:
          User, item, and duplicate_mask arrays.
        """
        items = np.concatenate([positive_items, negative_items], axis=1)

        # We pad the users and items here so that the duplicate mask calculation
        # will include padding. The metric function relies on all padded elements
        # except the positive being marked as duplicate to mask out padded points.
        if users.shape[0] < users_per_batch:
            pad_rows = users_per_batch - users.shape[0]
            padding = np.zeros(shape=(pad_rows, users.shape[1]), dtype=np.int32)
            users = np.concatenate([users, padding.astype(users.dtype)], axis=0)
            items = np.concatenate([items, padding.astype(items.dtype)], axis=0)

        duplicate_mask = stat_utils.mask_duplicates(items, axis=1).astype(np.float32)

        items[:, (0, -1)] = items[:, (-1, 0)]
        duplicate_mask[:, (0, -1)] = duplicate_mask[:, (-1, 0)]

        assert users.shape == items.shape == duplicate_mask.shape
        return users, items, duplicate_mask

    def _get_eval_item(self, index):
        """Get eval item"""
        low_index, high_index = index
        users = np.repeat(self._pos_users[low_index:high_index, np.newaxis],
                          1 + rconst.NUM_EVAL_NEGATIVES, axis=1)
        positive_items = self._pos_items[low_index:high_index, np.newaxis]
        negative_items = (self.lookup_negative_items(negative_users=users[:, :-1])
                          .reshape(-1, rconst.NUM_EVAL_NEGATIVES))

        users, items, duplicate_mask = self._assemble_eval_batch(
            users, positive_items, negative_items, self._eval_users_per_batch)

        users = np.reshape(users.flatten(), (self._batch_size, 1))  # (self._batch_size, 1), int32
        items = np.reshape(items.flatten(), (self._batch_size, 1))  # (self._batch_size, 1), int32
        duplicate_mask = np.reshape(duplicate_mask.flatten(), (self._batch_size, 1))  # (self._batch_size, 1), bool

        return users, items, duplicate_mask

    def __getitem__(self, index):
        """
        Get a batch of samples.
        """
        if self._is_training:
            return self._get_train_item(index)

        return self._get_eval_item(index)

    def __len__(self):
        """
        Return length of the dataset, i.e., the number of batches for an epoch
        """
        return self.dataset_len


class RandomSampler(Sampler):
    """
    A random sampler for dataset.
    """

    def __init__(self, pos_count, num_train_negatives, batch_size):
        self.pos_count = pos_count
        self._num_samples = (1 + num_train_negatives) * self.pos_count
        self._batch_size = batch_size
        self._num_batches = math.ceil(self._num_samples / self._batch_size)
        super().__init__(self._num_batches)

    def __iter__(self):
        """
        Return indices of all batches within an epoch.
        """
        indices = stat_utils.permutation((self._num_samples, stat_utils.random_int32()))

        batch_indices = [indices[x * self._batch_size:(x + 1) * self._batch_size] for x in range(self._num_batches)]

        # padding last batch indices if necessary
        if len(batch_indices) > 2 and len(batch_indices[-2]) != len(batch_indices[-1]):
            pad_nums = len(batch_indices[-2]) - len(batch_indices[-1])
            pad_indices = np.random.randint(0, self._num_samples, pad_nums)
            batch_indices[-1] = np.hstack((batch_indices[-1], pad_indices))

        return iter(batch_indices)


class DistributedSamplerOfTrain:
    """
    A distributed sampler for dataset.
    """

    def __init__(self, pos_count, num_train_negatives, batch_size, rank_id, rank_size):
        """
        Distributed sampler of training dataset.
        """
        self._num_samples = (1 + num_train_negatives) * pos_count
        self._rank_id = rank_id
        self._rank_size = rank_size
        self._batch_size = batch_size

        self._batchs_per_rank = int(math.ceil(self._num_samples / self._batch_size / rank_size))
        self._samples_per_rank = int(math.ceil(self._batchs_per_rank * self._batch_size))
        self._total_num_samples = self._samples_per_rank * self._rank_size

    def __iter__(self):
        """
        Returns the data after each sampling.
        """
        indices = stat_utils.permutation((self._num_samples, stat_utils.random_int32()))
        indices = indices.tolist()
        indices.extend(indices[:self._total_num_samples - len(indices)])
        indices = indices[self._rank_id:self._total_num_samples:self._rank_size]
        batch_indices = [indices[x * self._batch_size:(x + 1) * self._batch_size] for x in range(self._batchs_per_rank)]

        return iter(np.array(batch_indices))

    def __len__(self):
        """
        Returns the length after each sampling.
        """
        return self._batchs_per_rank


class SequenceSampler(Sampler):
    """
    A sequence sampler for dataset.
    """

    def __init__(self, eval_batch_size, num_users):
        self._eval_users_per_batch = int(
            eval_batch_size // (1 + rconst.NUM_EVAL_NEGATIVES))
        self._eval_elements_in_epoch = num_users * (1 + rconst.NUM_EVAL_NEGATIVES)
        self._eval_batches_per_epoch = self.count_batches(
            self._eval_elements_in_epoch, eval_batch_size)
        super().__init__(self._eval_batches_per_epoch)

    def __iter__(self):
        indices = [(x * self._eval_users_per_batch, (x + 1) * self._eval_users_per_batch)
                   for x in range(self._eval_batches_per_epoch)]

        # padding last batch indices if necessary
        if len(indices) > 2 and len(indices[-2]) != len(indices[-1]):
            pad_nums = len(indices[-2]) - len(indices[-1])
            pad_indices = np.random.randint(0, self._eval_elements_in_epoch, pad_nums)
            indices[-1] = np.hstack((indices[-1], pad_indices))

        return iter(indices)

    @staticmethod
    def count_batches(example_count, batch_size, batches_per_step=1):
        """Determine the number of batches, rounding up to fill all devices."""
        x = (example_count + batch_size - 1) // batch_size
        return (x + batches_per_step - 1) // batches_per_step * batches_per_step


class DistributedSamplerOfEval:
    """
    A distributed sampler for eval dataset.
    """

    def __init__(self, eval_batch_size, num_users, rank_id, rank_size):
        self._eval_users_per_batch = int(
            eval_batch_size // (1 + rconst.NUM_EVAL_NEGATIVES))
        self._eval_elements_in_epoch = num_users * (1 + rconst.NUM_EVAL_NEGATIVES)
        self._eval_batches_per_epoch = self.count_batches(
            self._eval_elements_in_epoch, eval_batch_size)

        self._rank_id = rank_id
        self._rank_size = rank_size
        self._eval_batch_size = eval_batch_size

        self._batchs_per_rank = int(math.ceil(self._eval_batches_per_epoch / rank_size))

    def __iter__(self):
        indices = [(x * self._eval_users_per_batch, (x + self._rank_id + 1) * self._eval_users_per_batch)
                   for x in range(self._batchs_per_rank)]

        return iter(np.array(indices))

    @staticmethod
    def count_batches(example_count, batch_size, batches_per_step=1):
        """Determine the number of batches, rounding up to fill all devices."""
        x = (example_count + batch_size - 1) // batch_size
        return (x + batches_per_step - 1) // batches_per_step * batches_per_step

    def __len__(self):
        return self._batchs_per_rank


def parse_eval_batch_size(eval_batch_size):
    """
    Parse eval batch size.
    """
    if eval_batch_size % (1 + rconst.NUM_EVAL_NEGATIVES):
        raise ValueError("Eval batch size {} is not divisible by {}".format(
            eval_batch_size, 1 + rconst.NUM_EVAL_NEGATIVES))
    return eval_batch_size


def create_dataset(test_train=True, data_dir='./dataset/', dataset='ml-1m', train_epochs=14, batch_size=256,
                   eval_batch_size=160000, num_neg=4, rank_id=None, rank_size=None):
    """
    Create NCF dataset.
    """
    data, num_users, num_items = load_data(data_dir, dataset)

    train_pos_users = data[rconst.TRAIN_USER_KEY]
    train_pos_items = data[rconst.TRAIN_ITEM_KEY]
    eval_pos_users = data[rconst.EVAL_USER_KEY]
    eval_pos_items = data[rconst.EVAL_ITEM_KEY]

    total_negatives, index_bounds, sorted_train_pos_items = \
        construct_lookup_variables(train_pos_users, train_pos_items, num_users)

    if test_train:
        print(train_pos_users, train_pos_items, num_users, num_items, batch_size, total_negatives, index_bounds,
              sorted_train_pos_items)
        dataset = NCFDataset(train_pos_users, train_pos_items, num_users, num_items, batch_size, total_negatives,
                             index_bounds, sorted_train_pos_items, num_neg)
        sampler = RandomSampler(train_pos_users.shape[0], num_neg, batch_size)
        if rank_id is not None and rank_size is not None:
            sampler = DistributedSamplerOfTrain(train_pos_users.shape[0], num_neg, batch_size, rank_id, rank_size)
        if dataset == 'ml-20m':
            ds = GeneratorDataset(dataset,
                                  column_names=[movielens.USER_COLUMN,
                                                movielens.ITEM_COLUMN,
                                                "labels",
                                                rconst.VALID_POINT_MASK],
                                  sampler=sampler, num_parallel_workers=32, python_multiprocessing=False)
        else:
            ds = GeneratorDataset(dataset,
                                  column_names=[movielens.USER_COLUMN,
                                                movielens.ITEM_COLUMN,
                                                "labels",
                                                rconst.VALID_POINT_MASK],
                                  sampler=sampler)

    else:
        eval_batch_size = parse_eval_batch_size(eval_batch_size=eval_batch_size)
        dataset = NCFDataset(eval_pos_users, eval_pos_items, num_users, num_items,
                             eval_batch_size, total_negatives, index_bounds,
                             sorted_train_pos_items, num_neg, is_training=False)
        sampler = SequenceSampler(eval_batch_size, num_users)

        ds = GeneratorDataset(dataset,
                              column_names=[movielens.USER_COLUMN,
                                            movielens.ITEM_COLUMN,
                                            rconst.DUPLICATE_MASK],
                              sampler=sampler)

    repeat_count = train_epochs if test_train else train_epochs + 1
    ds = ds.repeat(repeat_count)

    return ds, num_users, num_items
