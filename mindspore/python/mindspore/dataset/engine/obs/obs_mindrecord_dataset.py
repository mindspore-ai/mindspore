# Copyright 2022 Huawei Technologies Co., Ltd
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
The dataset module provide the internal Dataset API which load mindrecord files from OBS.
"""


import math
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.managers import SyncManager
import os
import queue
import random
import stat
import sys
import time

from mindspore import log as logger
from ..datasets import Shuffle
from ...core.config import set_seed


class _Manager(SyncManager):
    pass


def _get_manager():
    """ PriorityQueue that cross threads."""

    _Manager.register("PriorityQueue", queue.PriorityQueue)
    m = _Manager()
    m.start()
    return m


def _init_cache_and_working_queue(cache, q, shard_files, local_path):
    """
    Initialize the downloading queue and local cache which store the status of local dataset file.
    """

    from .util import init_cache_and_queue

    idx = 0
    for shard_file, _, _, is_full_dataset in shard_files:
        dataset_file = os.path.basename(shard_file)
        path = os.path.join(local_path, dataset_file)
        init_cache_and_queue(cache, q, path, shard_file,
                             idx, is_full_dataset, lock_file=dataset_file)
        idx += 1
    return cache, q


def _remove_unused_dataset(local_path, num_shards, shard_id, epoch_num):
    """ Rank(rank_id mod 8 equal to 0) remove all dataset files. """

    from .config_loader import config

    if not num_shards:
        return
    # if num_shards less than or equal to 8, assume that there is only one node(server) and
    # the dataset does not need to be removed.
    if num_shards <= 8 or shard_id % 8 != 0:
        return

    sync_dir = '/cache/sync_data/' + str(epoch_num)
    while True:
        if os.path.exists(sync_dir) and len(os.listdir(sync_dir)) >= min(num_shards - 1, 7):
            break
        time.sleep(config.WARMINGUP_TIME)
        logger.info("[{} FUNCTION] Shard: {} wait for other rank ready in epoch: {}.".format(
            sys._getframe().f_code.co_name, shard_id, epoch_num))  # pylint: disable=W0212

    files = os.listdir(local_path)
    for dataset_file in files:
        if dataset_file.endswith('.db'):
            continue
        dataset_path = os.path.join(local_path, dataset_file)
        os.remove(dataset_path)

    for ready_file in os.listdir(sync_dir):
        os.remove(os.path.join(sync_dir, ready_file))


def _wait_remove_datset(num_shards, shard_id, epoch_num):
    """ Rank(rank_id mod 8 not equal to 0) wait for removing dataset files. """

    from .config_loader import config

    if not num_shards:
        return
    if num_shards <= 8 or shard_id % 8 == 0:
        return

    sync_dir = '/cache/sync_data/' + str(epoch_num)

    if not os.path.exists(sync_dir):
        try:
            os.makedirs(sync_dir)
        except FileExistsError:
            pass

    sync_file = os.path.join(sync_dir, 'ready_' + str(shard_id))
    with open(sync_file, 'w') as f:
        f.write('ok')

    if os.path.exists(sync_file):
        os.chmod(sync_file, stat.S_IRUSR | stat.S_IWUSR)

    while True:
        if os.path.exists(sync_dir) and not os.listdir(sync_dir):
            break
        time.sleep(config.WARMINGUP_TIME)
        logger.info("[{} FUNCTION] Shard: {} wait for removing dataset files in epoch: {}.".format(
            sys._getframe().f_code.co_name, shard_id, epoch_num))  # pylint: disable=W0212


def _init_shard_files(dataset_files, shuffle, seed, num_shards, shard_id, shard_equal_rows,
                      size_per_shard, local_path, current_epoch):
    """ Calculate the dataset files required by each sharding and the corresponding index. """

    from .config_loader import config
    from .util import detect_all_meta_files, fetch_meta_files, make_dataset_tuple, make_shard_files, make_shard_samples

    shard_files = None
    if shuffle is False or shuffle == Shuffle.INFILE:
        pass
    else:
        set_seed(seed)
        random.shuffle(dataset_files)
    if num_shards:  # distributed training
        # As each sharding has the same number of samples, need to fetch all meta files.
        if shard_equal_rows:
            if size_per_shard is None:
                if shard_id % 8 == 0:
                    fetch_meta_files(dataset_files, local_path)
                else:
                    while detect_all_meta_files(dataset_files, local_path) is False:
                        time.sleep(config.WAIT_META_TIME)
            full_dataset_size, dataset_file_size_list = make_dataset_tuple(
                dataset_files, local_path)
            size_per_shard = math.ceil(full_dataset_size / num_shards)
            shard_files = make_shard_samples(
                dataset_file_size_list, size_per_shard, shard_id)
        else:
            shard_files = make_shard_files(dataset_files, num_shards, shard_id)
    else:
        shard_files = [(dataset_file, -1, -1, True)
                       for dataset_file in dataset_files]
    logger.info("[{} FUNCTION] Shard: {} expect dataset: {} in epoch: {}.".format(
        sys._getframe().f_code.co_name, shard_id, shard_files, current_epoch))  # pylint: disable=W0212
    return shard_files, size_per_shard


def _download_work(shard_id, current_idx, local_path, cache, q):
    """ daemon process in backend. """
    from .config_loader import config
    from .util import try_load_from_obs, get_used_disk_per

    while True:
        idx, dataset_file = q.get()
        used_disk = get_used_disk_per()
        while used_disk > float(config.DISK_THRESHOLD):
            logger.info("[{} FUNCTION] Used disk space is {}%, and the disk threshold is {}%.".format(
                sys._getframe().f_code.co_name, used_disk * 100,  # pylint: disable=W0212
                float(config.DISK_THRESHOLD)*100))
            retry_cnt = 0
            has_deleted = _delete_candidate_datasets(
                current_idx.value, idx, cache, q, local_path)
            while not has_deleted:
                if retry_cnt > config.MAX_RETRY:
                    logger.warning("Delete operation retries times {} has exceeded threshold {}, "
                                   "please clear enough disk space.".format(retry_cnt, config.MAX_RETRY))
                has_deleted = _delete_candidate_datasets(
                    current_idx.value, idx, cache, q, local_path)
                retry_cnt += 1
                time.sleep(config.RETRY_DELTA_TIME)
            used_disk = get_used_disk_per()

        logger.info("[{} FUNCTION] Shard: {} try to download: {}.".format(
            sys._getframe().f_code.co_name, shard_id, dataset_file))  # pylint: disable=W0212
        # update cache
        remote_path = os.path.dirname(dataset_file)
        dataset_file = os.path.basename(dataset_file)
        _, is_shared = cache[dataset_file]
        try_load_from_obs(remote_path, dataset_file, local_path)
        cache[dataset_file] = (idx, is_shared)
        logger.info("[{} FUNCTION] Shard: {} finish to download: {}.".format(
            sys._getframe().f_code.co_name, shard_id, dataset_file))  # pylint: disable=W0212


def _delete_candidate_datasets(current_idx, queue_top_idx, cache, q, local_path):
    """
    1. Try to delete all the datasets which have been loaded during the epoch.
    2. Otherwise, try to delete a low priority dataset in the epoch.
    3. As soon as the low priority data is deleted, it is placed in the download queue.
    """

    used_datasets = []
    low_priority_dataset = ''
    max_idx = -1
    delete = False
    for k, v in cache.items():
        idx, is_shared = v
        if is_shared is False and idx >= 0:
            if idx > max_idx:
                max_idx = idx
                low_priority_dataset = k
            if idx < current_idx:
                used_datasets.append(k)
    for used_dataset in used_datasets:
        dataset_path = os.path.join(local_path, used_dataset)
        if not os.path.exists(dataset_path):
            continue
        # update cache
        idx, is_shared = cache[used_dataset]
        cache[used_dataset] = (-1, is_shared)
        os.remove(dataset_path)
        delete = True
        logger.info("[{} FUNCTION] Delete used dataset file: {} and update the cache.".format(
            sys._getframe().f_code.co_name, used_dataset))  # pylint: disable=W0212

    if delete:
        return True
    if max_idx <= current_idx or max_idx <= queue_top_idx:
        return False
    dataset_path = os.path.join(local_path, low_priority_dataset)
    if not os.path.exists(dataset_path):
        return False
    # update cache
    idx, is_shared = cache[low_priority_dataset]
    cache[low_priority_dataset] = (-1, is_shared)
    os.remove(dataset_path)
    q.put((idx, low_priority_dataset))
    logger.info("[{} FUNCTION] Delete low priority dataset file: {} and update the cache.".format(
        sys._getframe().f_code.co_name, low_priority_dataset))  # pylint: disable=W0212
    return True


def _sync_up_for_obs_mindrecord_dataset(rank_id, current_epoch):
    """ Upload the synchronization file to OBS. """

    from .config_loader import config
    from .util import file_upload_to_obs

    sync_info = "download_dataset"
    job_id = os.environ.get('BATCH_JOB_ID', 'unknown')
    ready_file_name = sync_info + '_ready_' + str(rank_id) + '.txt'
    ready_dir = os.path.join(job_id, str(current_epoch) + "/")

    file_upload_to_obs(config.SYNC_OBS_PATH, ready_dir, ready_file_name)
    logger.info("[{} FUNCTION] Current rank:{}'s sync file:{} is ready for epoch:{}.".format(
        sys._getframe().f_code.co_name, rank_id, os.path.join(ready_dir, ready_file_name),  # pylint: disable=W0212
        current_epoch))


def sync_wait_for_dataset(rank_id, rank_size, current_epoch):
    """
    Wait util the dataset files required by all devices are downloaded.

    Note:
        It should be used together with :class:`mindspore.dataset.OBSMindDataset` and
        be called before each epoch.

    Args:
        rank_id(int): Rank ID of the device.
        rank_size(int): Rank size.
        current_epoch(int): Number of current epochs.

    Examples:
        >>> # Create a synchronization callback
        >>> import mindspore as ms
        >>> from mindspore.dataset import sync_wait_for_dataset
        >>>
        >>> class SyncForDataset(ms.Callback):
        ...     def __init__(self):
        ...         super(SyncForDataset, self).__init__()
        ...     def epoch_begin(self, run_context):
        ...         cb_params = run_context.original_args()
        ...         epoch_num = cb_params.cur_epoch_num
        ...         sync_wait_for_dataset(rank_id, rank_size, epoch_num)

    """

    from .config_loader import config
    from .util import obsClient, get_bucket_and_key

    bucket_name, object_key = get_bucket_and_key(config.SYNC_OBS_PATH)

    job_id = os.environ.get('BATCH_JOB_ID', 'unknown')
    ready_dir = os.path.join(object_key, job_id, str(current_epoch) + "/")

    success = False
    while True:
        if success:
            break
        try:
            # no guarantee that the dir is included.
            resp = obsClient.listObjects(bucket_name, prefix=ready_dir)
            if resp.status < 300:
                ready_num = 0
                for content in resp.body.contents:
                    if content.key.endswith(".txt"):
                        ready_num += 1
                if ready_num >= rank_size:
                    success = True
            else:
                logger.warning("[{} FUNCTION] OBS SDK errorCode:{}, errMsg: {}.".format(
                    sys._getframe(), resp.errorCode, resp.errorMessage))  # pylint: disable=W0212
        except Exception:  # pylint: disable=W0703
            import traceback
            logger.error(traceback.format_exc())
        time.sleep(config.RETRY_DELTA_TIME)
        logger.info("[{} FUNCTION] Waiting for sync dir:{} and current_rank:{}, total_rank:{}, "
                    "ready_rank:{} in epoch:{}.".format(sys._getframe().f_code.co_name,  # pylint: disable=W0212
                                                        ready_dir, rank_id, rank_size, ready_num, current_epoch))
    logger.info("[{} FUNCTION] Succeed to sync dir:{} and begin epoch:{}.".format(
        sys._getframe().f_code.co_name, ready_dir, current_epoch))  # pylint: disable=W0212


def _sync_for_obs_mindrecord_dataset(worker, shard_files, cache, num_shards, shard_id, current_epoch):
    """ Synchronize all shardings. """

    from .config_loader import config

    while True:
        if worker.ready():
            worker.get()
        dataset, _, _, _ = shard_files[-1]
        current_dataset = os.path.basename(dataset)
        hit_cache = cache[current_dataset][0]
        if hit_cache >= 0:  # hit cache
            logger.info("[{} FUNCTION] Current_rank:{} has download:{} for epoch:{}.".format(
                sys._getframe().f_code.co_name, shard_id, dataset, current_epoch))  # pylint: disable=W0212
            _sync_up_for_obs_mindrecord_dataset(shard_id, current_epoch)
            break
        time.sleep(config.WARMINGUP_TIME)
        logger.info("[{} FUNCTION] Current_rank:{} wait for downloading:{} in epoch:{}.".format(
            sys._getframe().f_code.co_name, shard_id, dataset, current_epoch))  # pylint: disable=W0212
    sync_wait_for_dataset(shard_id, num_shards, current_epoch)


class MindRecordFromOBS:
    """ Internal class which load remote dataset files from OBS. """

    def __init__(self, dataset_files, columns_list, shuffle, num_shards, shard_id, shard_equal_rows, local_path):
        self._dataset_files = dataset_files
        self._columns_list = columns_list

        self._num_shards = num_shards
        self._shard_id = shard_id

        self._shard_equal_rows = shard_equal_rows
        self._local_path = os.path.realpath(local_path)

        self._shuffle = Shuffle.GLOBAL if shuffle is True else shuffle
        from .config_loader import config
        self._epoch_seed = config.SEED
        self._file_seed = config.SEED
        self._size_per_shard = None
        self._curr_epoch = 1
        self._curr_step = 1
        self._shard_files, self._size_per_shard = _init_shard_files(self._dataset_files, self._shuffle,
                                                                    self._epoch_seed, self._num_shards, self._shard_id,
                                                                    self._shard_equal_rows, self._size_per_shard,
                                                                    self._local_path, self._curr_epoch)

        m = _get_manager()
        self._queue = m.PriorityQueue()
        self._cache = m.dict()

        self._index = 0
        self._current_idx = m.Value('i', self._index)

        self._cache, self._queue = _init_cache_and_working_queue(
            self._cache, self._queue, self._shard_files, self._local_path)

        self._index = 0
        self._first_epoch = True
        self._iteration = None
        self._cache_miss_times = 0

        self._pool = ThreadPool(processes=1)
        self._worker = self._pool.apply_async(
            _download_work, (self._shard_id, self._current_idx, self._local_path, self._cache, self._queue))
        _sync_for_obs_mindrecord_dataset(
            self._worker, self._shard_files, self._cache, self._num_shards, self._shard_id, self._curr_epoch)

    def __next__(self):
        from .config_loader import config
        from ..datasets_standard_format import MindDataset
        from .util import make_sampler

        if self._iteration:
            try:
                self._curr_step += 1
                return next(self._iteration)
            except StopIteration:
                self._index += 1
                self._current_idx.value = self._index

                self._iteration = None
                if self._index >= len(self._shard_files):
                    self._first_epoch = False
                    self._curr_epoch += 1
                    self._curr_step = 0
                    raise StopIteration
                return next(self)
        else:
            f, start, end, is_full_dataset = self._shard_files[self._index]
            current_dataset = os.path.basename(f)
            hit_cache = self._cache[current_dataset][0]
            if hit_cache >= 0:  # hit cache
                self._cache_miss_times = 0
                # launch pipeline

                set_seed(self._file_seed)
                sampler = make_sampler(
                    self._shuffle, is_full_dataset, start, end)
                self._file_seed += 1
                path = os.path.join(self._local_path, current_dataset)
                logger.info("[{} FUNCTION] Shard:{} start to load dataset:{} in epoch:{}.".format(
                    sys._getframe().f_code.co_name, self._shard_id, path, self._curr_epoch))  # pylint: disable=W0212
                self._iteration = MindDataset(dataset_files=[path], columns_list=self._columns_list, sampler=sampler,
                                              shuffle=None).create_tuple_iterator(num_epochs=1, output_numpy=True)

            else:
                # cache miss
                self._cache_miss_times += 1
                logger.info("[{} FUNCTION]  Cache miss in shard {} for times {}, expect dataset {}.".format(
                    sys._getframe().f_code.co_name, self._shard_id, self._cache_miss_times,  # pylint: disable=W0212
                    current_dataset))
                time.sleep(self._cache_miss_times * config.WAIT_STEP_TIME)
            return next(self)

    def __iter__(self):
        if self._first_epoch:
            self._index = 0
            self._current_idx.value = self._index
            self._iteration = None
            return self
        self._index = 0
        self._current_idx.value = self._index

        self._epoch_seed += 1
        self._iteration = None
        self._shard_files, self._size_per_shard = _init_shard_files(self._dataset_files, self._shuffle,
                                                                    self._epoch_seed, self._num_shards, self._shard_id,
                                                                    self._shard_equal_rows, self._size_per_shard,
                                                                    self._local_path, self._curr_epoch)
        self._cache.clear()
        # reset queue
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        _remove_unused_dataset(
            self._local_path, self._num_shards, self._shard_id, self._curr_epoch)
        _wait_remove_datset(self._num_shards, self._shard_id, self._curr_epoch)

        self._cache, self._queue = _init_cache_and_working_queue(
            self._cache, self._queue, self._shard_files, self._local_path)
        _sync_for_obs_mindrecord_dataset(self._worker,
                                         self._shard_files, self._cache,
                                         self._num_shards, self._shard_id, self._curr_epoch)
        return self

    def __len__(self):
        from .util import fetch_meta_files, make_dataset_tuple

        if self._size_per_shard is not None:
            return self._size_per_shard
        dataset_files = []
        for dataset_file, _, _, _ in self._shard_files:
            dataset_files.append(dataset_file)
        fetch_meta_files(dataset_files, self._local_path)
        self._size_per_shard, _ = make_dataset_tuple(
            dataset_files, self._local_path)
        return len(self)

    def get_col_names(self):
        """ Get column names of MindRecord format dataset."""

        from ..datasets_standard_format import MindDataset

        target_dataset = None
        while target_dataset is None:
            for f, _, _, _ in self._shard_files:
                current_dataset = os.path.basename(f)
                if self._cache[current_dataset][0] >= 0:
                    target_dataset = current_dataset
        path = os.path.join(self._local_path, target_dataset)
        _iteration = MindDataset(dataset_files=[path], shuffle=False)
        return _iteration.get_col_names()
