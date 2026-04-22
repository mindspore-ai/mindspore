# Copyright 2022-2026 Huawei Technologies Co., Ltd
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
import os
import queue
import random
import stat
import sys
import time
import traceback
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.managers import SyncManager

from mindspore import log as logger

from ...core.config import set_seed
from ..samplers import Shuffle
from .util import (
    OBSRuntime,
    detect_all_meta_files,
    get_bucket_and_key,
    init_cache_and_queue,
    make_dataset_tuple,
    make_sampler,
    make_shard_files,
    make_shard_samples,
)


class _Manager(SyncManager):
    pass


# pylint: disable=consider-using-with
def _get_manager():
    """PriorityQueue that cross threads."""

    _Manager.register("PriorityQueue", queue.PriorityQueue)
    # The manager instance must outlive this helper and be returned to the caller.
    m = _Manager()
    m.start()
    return m


# pylint: enable=consider-using-with


def _init_cache_and_working_queue(cache, q, shard_files, local_path):
    """
    Initialize the downloading queue and local cache which store the status of local dataset file.
    """

    idx = 0
    for shard_file, _, _, is_full_dataset in shard_files:
        dataset_file = os.path.basename(shard_file)
        path = os.path.join(local_path, dataset_file)
        init_cache_and_queue(cache, q, path, shard_file, idx, is_full_dataset, lock_file=dataset_file)
        idx += 1
    return cache, q


def _current_func_name():
    """Return caller function name for log messages."""
    return sys._getframe(1).f_code.co_name  # pylint: disable=protected-access


def _remove_unused_dataset(local_path, num_shards, shard_id, epoch_num, obs_runtime):
    """Rank(rank_id mod 8 equal to 0) remove all dataset files."""
    if not num_shards:
        return
    # if num_shards less than or equal to 8, assume that there is only one node(server) and
    # the dataset does not need to be removed.
    if num_shards <= 8 or shard_id % 8 != 0:
        return

    sync_dir = "/cache/sync_data/" + str(epoch_num)
    while True:
        if os.path.exists(sync_dir) and len(os.listdir(sync_dir)) >= min(num_shards - 1, 7):
            break
        time.sleep(obs_runtime.config.WARMINGUP_TIME)
        logger.info(
            f"[{_current_func_name()} FUNCTION] Shard: {shard_id} wait for other rank ready in epoch: {epoch_num}."
        )

    local_path_fd = os.open(local_path, os.O_RDONLY)
    try:
        files = os.listdir(local_path)
        for dataset_file in files:
            if dataset_file.endswith(".db"):
                continue
            try:
                os.unlink(dataset_file, dir_fd=local_path_fd)
            except FileNotFoundError:
                logger.warning(
                    f"[{_current_func_name()} FUNCTION] Dataset file: {dataset_file} was removed "
                    f"before cleanup in path: {local_path}."
                )
            except IsADirectoryError:
                logger.warning(
                    f"[{_current_func_name()} FUNCTION] Skip directory during dataset cleanup: {dataset_file}."
                )
    finally:
        os.close(local_path_fd)

    for ready_file in os.listdir(sync_dir):
        os.remove(os.path.join(sync_dir, ready_file))


def _wait_remove_datset(num_shards, shard_id, epoch_num, obs_runtime):
    """Rank(rank_id mod 8 not equal to 0) wait for removing dataset files."""
    if not num_shards:
        return
    if num_shards <= 8 or shard_id % 8 == 0:
        return

    sync_dir = "/cache/sync_data/" + str(epoch_num)

    if not os.path.exists(sync_dir):
        try:
            os.makedirs(sync_dir)
        except FileExistsError:
            pass

    sync_file = os.path.join(sync_dir, "ready_" + str(shard_id))
    with open(sync_file, "w", encoding="utf-8") as f:
        f.write("ok")

    if os.path.exists(sync_file):
        os.chmod(sync_file, stat.S_IRUSR | stat.S_IWUSR)

    while True:
        if os.path.exists(sync_dir) and not os.listdir(sync_dir):
            break
        time.sleep(obs_runtime.config.runtime_config.WARMINGUP_TIME)
        logger.info(
            f"[{_current_func_name()} FUNCTION] Shard: {shard_id} wait for removing dataset files "
            f"in epoch: {epoch_num}."
        )


def _init_shard_files(
    dataset_files,
    shuffle,
    seed,
    num_shards,
    shard_id,
    shard_equal_rows,
    size_per_shard,
    local_path,
    current_epoch,
    obs_runtime,
):
    """Calculate the dataset files required by each sharding and the corresponding index."""
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
                    obs_runtime.fetch_meta_files(dataset_files, local_path)
                else:
                    while detect_all_meta_files(dataset_files, local_path) is False:
                        time.sleep(obs_runtime.config.WAIT_META_TIME)
            full_dataset_size, dataset_file_size_list = make_dataset_tuple(dataset_files, local_path)
            size_per_shard = math.ceil(full_dataset_size / num_shards)
            shard_files = make_shard_samples(dataset_file_size_list, size_per_shard, shard_id)
        else:
            shard_files = make_shard_files(dataset_files, num_shards, shard_id)
    else:
        shard_files = [(dataset_file, -1, -1, True) for dataset_file in dataset_files]
    logger.info(
        f"[{_current_func_name()} FUNCTION] Shard: {shard_id} expect dataset: {shard_files} in epoch: {current_epoch}."
    )
    return shard_files, size_per_shard


def _download_work(shard_id, current_idx, local_path, cache, q, obs_runtime):
    """daemon process in backend."""
    runtime_config = obs_runtime.config
    while True:
        idx, dataset_file = q.get()
        used_disk = obs_runtime.get_used_disk_per()
        while used_disk > float(runtime_config.DISK_THRESHOLD):
            logger.info(
                f"[{_current_func_name()} FUNCTION] Used disk space is {used_disk * 100}%, "
                f"and the disk threshold is {float(runtime_config.DISK_THRESHOLD) * 100}%."
            )
            retry_cnt = 0
            has_deleted = _delete_candidate_datasets(current_idx.value, idx, cache, q, local_path)
            while not has_deleted:
                if retry_cnt > runtime_config.MAX_RETRY:
                    logger.warning(
                        f"Delete operation retries times {retry_cnt} has exceeded threshold "
                        f"{runtime_config.MAX_RETRY}, please clear enough disk space."
                    )
                has_deleted = _delete_candidate_datasets(current_idx.value, idx, cache, q, local_path)
                retry_cnt += 1
                time.sleep(runtime_config.RETRY_DELTA_TIME)
            used_disk = obs_runtime.get_used_disk_per()

        logger.info(f"[{_current_func_name()} FUNCTION] Shard: {shard_id} try to download: {dataset_file}.")
        # update cache
        remote_path = os.path.dirname(dataset_file)
        dataset_file = os.path.basename(dataset_file)
        _, is_shared = cache[dataset_file]
        obs_runtime.try_load_from_obs(remote_path, dataset_file, local_path)
        cache[dataset_file] = (idx, is_shared)
        logger.info(f"[{_current_func_name()} FUNCTION] Shard: {shard_id} finish to download: {dataset_file}.")


def _delete_candidate_datasets(current_idx, queue_top_idx, cache, q, local_path):
    """
    1. Try to delete all the datasets which have been loaded during the epoch.
    2. Otherwise, try to delete a low priority dataset in the epoch.
    3. As soon as the low priority data is deleted, it is placed in the download queue.
    """

    used_datasets = []
    low_priority_dataset = ""
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
        logger.info(f"[{_current_func_name()} FUNCTION] Delete used dataset file: {used_dataset} and update the cache.")

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
    logger.info(
        f"[{_current_func_name()} FUNCTION] Delete low priority dataset file: {low_priority_dataset} "
        f"and update the cache."
    )
    return True


def _sync_up_for_obs_mindrecord_dataset(rank_id, current_epoch, obs_runtime):
    """Upload the synchronization file to OBS."""
    runtime_config = obs_runtime.config
    sync_info = "download_dataset"
    job_id = os.environ.get("BATCH_JOB_ID", "unknown")
    ready_file_name = sync_info + "_ready_" + str(rank_id) + ".txt"
    ready_dir = os.path.join(job_id, str(current_epoch) + "/")

    obs_runtime.file_upload_to_obs(runtime_config.SYNC_OBS_PATH, ready_dir, ready_file_name)
    logger.info(
        f"[{_current_func_name()} FUNCTION] Current rank:{rank_id}'s sync file:"
        f"{os.path.join(ready_dir, ready_file_name)} is ready for epoch:{current_epoch}."
    )


def sync_wait_for_dataset(rank_id, rank_size, current_epoch, obs_runtime):
    """
    Wait until the dataset files required by all devices are downloaded.

    Note:
        It should be used together with :class:`mindspore.dataset.OBSMindDataset` and
        be called before each epoch.

    Args:
        rank_id (int): Rank ID of the device.
        rank_size (int): Rank size.
        current_epoch (int): Number of current epochs.
        obs_runtime (OBSRuntime): OBS runtime from the corresponding OBSMindDataset instance.

    Examples:
        >>> # Create a synchronization callback
        >>> import mindspore as ms
        >>> from mindspore.dataset import sync_wait_for_dataset
        >>>
        >>> class SyncForDataset(ms.Callback):
        ...     def __init__(self):
        ...         super(SyncForDataset, self).__init__()
        ...
        ...     def epoch_begin(self, run_context):
        ...         cb_params = run_context.original_args()
        ...         epoch_num = cb_params.cur_epoch_num
        ...         sync_wait_for_dataset(rank_id, rank_size, epoch_num, obs_runtime=dataset.source.get_obs_runtime())

    """

    runtime_config = obs_runtime.config
    bucket_name, object_key = get_bucket_and_key(runtime_config.SYNC_OBS_PATH)
    obs_client = obs_runtime.get_obs_client()

    job_id = os.environ.get("BATCH_JOB_ID", "unknown")
    ready_dir = os.path.join(object_key, job_id, str(current_epoch) + "/")

    success = False
    ready_num = 0
    while True:
        if success:
            break
        try:
            # no guarantee that the dir is included.
            resp = obs_client.listObjects(bucket_name, prefix=ready_dir)
            if resp.status < 300:
                ready_num = 0
                for content in resp.body.contents:
                    if content.key.endswith(".txt"):
                        ready_num += 1
                if ready_num >= rank_size:
                    success = True
            else:
                logger.warning(
                    f"[{_current_func_name()} FUNCTION] OBS SDK errorCode:{resp.errorCode}, "
                    f"errMsg: {resp.errorMessage}."
                )
        except Exception:  # pylint: disable=W0703
            logger.error(traceback.format_exc())
        time.sleep(runtime_config.RETRY_DELTA_TIME)
        logger.info(
            f"[{_current_func_name()} FUNCTION] Waiting for sync dir:{ready_dir} and "
            f"current_rank:{rank_id}, total_rank:{rank_size}, ready_rank:{ready_num} in epoch:{current_epoch}."
        )
    logger.info(f"[{_current_func_name()} FUNCTION] Succeed to sync dir:{ready_dir} and begin epoch:{current_epoch}.")


def _sync_for_obs_mindrecord_dataset(worker, shard_files, cache, num_shards, shard_id, current_epoch, obs_runtime):
    """Synchronize all shardings."""
    while True:
        if worker.ready():
            worker.get()
        dataset, _, _, _ = shard_files[-1]
        current_dataset = os.path.basename(dataset)
        hit_cache = cache[current_dataset][0]
        if hit_cache >= 0:  # hit cache
            logger.info(
                f"[{_current_func_name()} FUNCTION] Current_rank:{shard_id} has download:"
                f"{dataset} for epoch:{current_epoch}."
            )
            _sync_up_for_obs_mindrecord_dataset(shard_id, current_epoch, obs_runtime=obs_runtime)
            break
        time.sleep(obs_runtime.config.WARMINGUP_TIME)
        logger.info(
            f"[{_current_func_name()} FUNCTION] Current_rank:{shard_id} wait for downloading:"
            f"{dataset} in epoch:{current_epoch}."
        )
    sync_wait_for_dataset(shard_id, num_shards, current_epoch, obs_runtime=obs_runtime)


class MindRecordFromOBS:
    """Internal class which load remote dataset files from OBS."""

    def __init__(self, dataset_files, columns_list, shuffle, num_shards, shard_id, shard_equal_rows, obs_config):
        self._dataset_files = dataset_files
        self._columns_list = columns_list

        self._num_shards = num_shards
        self._shard_id = shard_id

        self._shard_equal_rows = shard_equal_rows
        self._shuffle = Shuffle.GLOBAL if shuffle is True else shuffle
        self._obs_config = obs_config
        self._obs_runtime = OBSRuntime(self._obs_config)
        self._local_path = os.path.realpath(self._obs_config.DATASET_LOCAL_PATH)
        self._epoch_seed = self._obs_config.SEED
        self._file_seed = self._obs_config.SEED
        self._size_per_shard = None
        self._curr_epoch = 1
        self._curr_step = 1
        self._shard_files, self._size_per_shard = _init_shard_files(
            self._dataset_files,
            self._shuffle,
            self._epoch_seed,
            self._num_shards,
            self._shard_id,
            self._shard_equal_rows,
            self._size_per_shard,
            self._local_path,
            self._curr_epoch,
            self._obs_runtime,
        )

        m = _get_manager()
        self._queue = m.PriorityQueue()
        self._cache = m.dict()

        self._index = 0
        self._current_idx = m.Value("i", self._index)

        self._cache, self._queue = _init_cache_and_working_queue(
            self._cache, self._queue, self._shard_files, self._local_path
        )

        self._index = 0
        self._first_epoch = True
        self._iteration = None
        self._cache_miss_times = 0

        self._pool = ThreadPool(processes=1)
        self._worker = self._pool.apply_async(
            _download_work,
            (self._shard_id, self._current_idx, self._local_path, self._cache, self._queue, self._obs_runtime),
        )
        _sync_for_obs_mindrecord_dataset(
            self._worker,
            self._shard_files,
            self._cache,
            self._num_shards,
            self._shard_id,
            self._curr_epoch,
            self._obs_runtime,
        )

    def __next__(self):
        from ..datasets_standard_format import MindDataset

        if self._iteration:
            try:
                self._curr_step += 1
                return next(self._iteration)
            except StopIteration as exc:
                self._index += 1
                self._current_idx.value = self._index

                self._iteration = None
                if self._index >= len(self._shard_files):
                    self._first_epoch = False
                    self._curr_epoch += 1
                    self._curr_step = 0
                    raise StopIteration from exc
                return next(self)
        else:
            f, start, end, is_full_dataset = self._shard_files[self._index]
            current_dataset = os.path.basename(f)
            hit_cache = self._cache[current_dataset][0]
            if hit_cache >= 0:  # hit cache
                self._cache_miss_times = 0
                # launch pipeline

                set_seed(self._file_seed)
                sampler = make_sampler(self._shuffle, is_full_dataset, start, end)
                self._file_seed += 1
                path = os.path.join(self._local_path, current_dataset)
                logger.info(
                    f"[{_current_func_name()} FUNCTION] Shard:{self._shard_id} start to load dataset:"
                    f"{path} in epoch:{self._curr_epoch}."
                )
                self._iteration = MindDataset(
                    dataset_files=[path], columns_list=self._columns_list, sampler=sampler, shuffle=None
                ).create_tuple_iterator(num_epochs=1, output_numpy=True)

            else:
                # cache miss
                self._cache_miss_times += 1
                logger.info(
                    f"[{_current_func_name()} FUNCTION]  Cache miss in shard {self._shard_id} for "
                    f"times {self._cache_miss_times}, expect dataset {current_dataset}."
                )
                time.sleep(self._cache_miss_times * self._obs_config.WAIT_STEP_TIME)
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
        self._shard_files, self._size_per_shard = _init_shard_files(
            self._dataset_files,
            self._shuffle,
            self._epoch_seed,
            self._num_shards,
            self._shard_id,
            self._shard_equal_rows,
            self._size_per_shard,
            self._local_path,
            self._curr_epoch,
            self._obs_runtime,
        )
        self._cache.clear()
        # reset queue
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        _remove_unused_dataset(self._local_path, self._num_shards, self._shard_id, self._curr_epoch, self._obs_runtime)
        _wait_remove_datset(self._num_shards, self._shard_id, self._curr_epoch, self._obs_runtime)

        self._cache, self._queue = _init_cache_and_working_queue(
            self._cache, self._queue, self._shard_files, self._local_path
        )
        _sync_for_obs_mindrecord_dataset(
            self._worker,
            self._shard_files,
            self._cache,
            self._num_shards,
            self._shard_id,
            self._curr_epoch,
            self._obs_runtime,
        )
        return self

    def __len__(self):
        if self._size_per_shard is not None:
            return self._size_per_shard
        dataset_files = []
        for dataset_file, _, _, _ in self._shard_files:
            dataset_files.append(dataset_file)
        self._obs_runtime.fetch_meta_files(dataset_files, self._local_path)
        self._size_per_shard, _ = make_dataset_tuple(dataset_files, self._local_path)
        return self._size_per_shard

    def get_col_names(self):
        """Get column names of MindRecord format dataset."""
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

    def get_obs_runtime(self):
        """Get runtime object bound to this dataset instance."""
        return self._obs_runtime

    def close(self):
        if hasattr(self, "_pool") and self._pool:
            self._pool.terminate()
            self._pool = None

    def __del__(self):
        self.close()
