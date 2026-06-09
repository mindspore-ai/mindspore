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
This dataset module provides internal utility function for OBSMindDataset API.
"""

import hashlib
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
import traceback
from functools import wraps

from mindspore import log as logger

from ..datasets import Shuffle
from ..samplers import RandomSampler, SequentialSampler, SubsetRandomSampler, SubsetSampler

if os.name == "nt":
    import msvcrt  # pylint: disable=import-error
else:
    import fcntl
try:
    from obs import ObsClient
except ImportError:
    ObsClient = None

LOCK_FILE_DIR = os.path.join(tempfile.gettempdir(), "mindspore_dataset_locks")


def _lock_file(fd):
    """Lock file descriptor exclusively on current platform."""
    if os.name == "nt":
        fd.seek(0)
        fd.write("0")
        fd.flush()
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_file(fd):
    """Release file lock on current platform."""
    if os.name == "nt":
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _retry_execute(max_retry, retry_delta_time, func_name, execute_fn):
    """Execute a callback with retries."""
    retry_cnt = 0
    while True:
        try:
            if retry_cnt >= max_retry:
                err_msg = f"Function {func_name} has retried for {retry_cnt} times, please check error above."
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            return execute_fn()
        except RuntimeError:
            raise
        except Exception:  # pylint: disable=broad-except
            retry_cnt += 1

            logger.error(traceback.format_exc())
            time.sleep(retry_delta_time)


def exclusive_lock(func):
    """Decorator that execute func under exclusive lock."""
    func_name = func.__name__

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            lock_file = _get_lock_file_path(kwargs["lock_file"])
        except KeyError as exc:
            raise RuntimeError(f"Lock file can not found in function {func_name}.") from exc
        with open(lock_file, "w", encoding="utf-8") as fd:
            try:
                _lock_file(fd)
                result = func(*args, **kwargs)
            finally:
                _unlock_file(fd)
        if os.path.exists(lock_file):
            os.chmod(lock_file, stat.S_IRUSR | stat.S_IWUSR)
        return result

    return wrapped_func


class OBSRuntime:
    """Runtime wrapper that keeps OBS config and client creation logic instance-local."""

    def __init__(self, obs_config):
        self.config = obs_config
        self.obs_client = None

    def get_obs_client(self):
        """Create an OBS client for this runtime."""
        if ObsClient is None:
            raise RuntimeError("OBS SDK is not installed. Please install dependency 'obs'.")
        if self.obs_client is None:
            self.obs_client = ObsClient(
                access_key_id=self.config.AK, secret_access_key=self.config.SK, server=self.config.SERVER
            )
        return self.obs_client

    def close(self):
        """Close the cached OBS client."""
        if self.obs_client is not None:
            self.obs_client.close()
            self.obs_client = None

    def __del__(self):
        self.close()

    def get_used_disk_per(self):
        """Get the disk usage of working directory."""
        working_path = self.config.WORKING_PATH
        if os.path.exists(working_path) and not os.access(working_path, os.R_OK | os.W_OK | os.X_OK):
            err_msg = (
                f"Permission denied:'{working_path}'. Suggestion: "
                f"1) It is recommended to manually create the {working_path} directory and add read/write "
                "permissions to it. "
                "2) If you can't create, we suggest you download MindRecord manually and "
                "read it using the MindDataset interface."
            )
            raise RuntimeError(err_msg)

        if not os.path.exists(working_path):
            try:
                os.makedirs(working_path)
            except FileExistsError:
                pass
            except PermissionError as e:
                err_msg = (
                    f"{e}. Suggestion: "
                    f"1) It is recommended to manually create the {working_path} directory and add read/write "
                    "permissions to it. "
                    "2) If you can't create, we suggest you download MindRecord manually and "
                    "read it using the MindDataset interface."
                )
                raise RuntimeError(err_msg) from e

        total, used, _ = shutil.disk_usage(working_path)
        return used / total

    def try_load_from_obs(self, remote_path, dataset_file, local_path):
        """Download one dataset file and its metadata if missing in local cache."""
        if not os.path.exists(os.path.join(local_path, dataset_file)):
            self._download_file(remote_path, dataset_file, local_path, lock_file=dataset_file)
        meta_file = dataset_file + ".db"
        if not os.path.exists(os.path.join(local_path, meta_file)):
            self._download_file(remote_path, meta_file, local_path, lock_file=meta_file)

    def fetch_meta_files(self, meta_files, local_path):
        """Download all metadata files from OBS."""
        for df in meta_files:
            dataset_file = os.path.basename(df)
            meta_file = dataset_file + ".db"
            remote_path = os.path.dirname(df)
            self._download_file(remote_path, meta_file, local_path, lock_file=meta_file)

    def file_upload_to_obs(self, obs_path, sync_dir, ready_file_name):
        """Upload one synchronization file to OBS."""
        bucket_name, object_key = get_bucket_and_key(obs_path)
        obs_client = self.get_obs_client()

        def _upload():
            if not object_key:
                resp = obs_client.headBucket(bucket_name)
            else:
                normalized_key = object_key
                if not normalized_key.endswith("/"):
                    normalized_key += "/"
                resp = obs_client.getObjectMetadata(bucket_name, normalized_key)

            if resp.status < 300:
                logger.debug(f"[{sys._getframe()} FUNCTION] OBS requestId: {resp.requestId}.")  # pylint: disable=W0212
            else:
                raise RuntimeError(
                    f"Directory/Bucket used for synchronization {obs_path} is not found in OBS, "
                    "please create it on OBS first."
                )

            remote_dir = os.path.join(object_key, sync_dir)
            resp = obs_client.putContent(bucket_name, remote_dir, content=None)
            if resp.status < 300:
                logger.debug(f"[{sys._getframe()} FUNCTION] OBS requestId: {resp.requestId}.")  # pylint: disable=W0212
            else:
                raise Exception(f"OBS SDK errorCode:{resp.errorCode}, errMsg: {resp.errorMessage}.")
            resp = obs_client.putContent(bucket_name, os.path.join(remote_dir, ready_file_name), content="OK")
            if resp.status < 300:
                logger.debug(f"[{sys._getframe()} FUNCTION] OBS requestId: {resp.requestId}.")  # pylint: disable=W0212
            else:
                raise Exception(f"OBS SDK errorCode:{resp.errorCode}, errMsg: {resp.errorMessage}.")

        return _retry_execute(self.config.MAX_RETRY, self.config.RETRY_DELTA_TIME, "file_upload_to_obs", _upload)

    def _check_file_exists_in_obs(self, obs_path):
        """Detect that file exists in OBS."""
        bucket_name, object_key = get_bucket_and_key(obs_path)
        obs_client = self.get_obs_client()

        def _check():
            try:
                resp = obs_client.getObjectMetadata(bucket_name, object_key)
            except ConnectionRefusedError as exc:
                err_msg = f"Failed to connect to OBS, please check OBS sever {obs_client.server}:{obs_client.port}."
                logger.error(err_msg)
                raise RuntimeError(err_msg) from exc

            if resp.status < 300:
                logger.debug(f"[{sys._getframe()} FUNCTION] OBS requestId: {resp.requestId}.")  # pylint: disable=W0212
            elif resp.status == 403:
                err_msg = "OBS access is Forbidden, please check AK or SK."
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            else:
                err_msg = f"File {obs_path} not found in OBS, please check again."
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        return _retry_execute(self.config.MAX_RETRY, self.config.RETRY_DELTA_TIME, "_check_file_exists_in_obs", _check)

    def _file_download_from_obs(self, obs_path, local_path):
        """Download one file from OBS."""
        bucket_name, object_key = get_bucket_and_key(obs_path)
        obs_client = self.get_obs_client()

        def _download():
            resp = obs_client.downloadFile(
                bucket_name, object_key, local_path, self.config.PART_SIZE, self.config.TASK_NUM, True
            )
            if resp.status < 300:
                logger.debug(f"[{sys._getframe()} FUNCTION] OBS requestId: {resp.requestId}.")  # pylint: disable=W0212
            else:
                raise Exception(f"OBS SDK errorCode:{resp.errorCode}, errMsg: {resp.errorMessage}.")

        return _retry_execute(self.config.MAX_RETRY, self.config.RETRY_DELTA_TIME, "_file_download_from_obs", _download)

    @exclusive_lock
    def _download_file(self, remote_path, object_name, des_path, lock_file="tmp"):
        """
        Download file from OBS exclusively.

        Args:
            remote_path (str): OBS directory path which dataset file is stored.
            object_name (str): Name of dataset file.
            des_path (str): Local directory path which dataset file is stored.
            lock_file (str): File name to lock.
        """

        local_path = os.path.join(des_path, object_name)
        if os.path.exists(local_path):
            return
        if not os.path.exists(des_path):
            os.makedirs(des_path)
        obs_path = os.path.join(remote_path, object_name)

        self._check_file_exists_in_obs(obs_path)
        self._file_download_from_obs(obs_path, local_path)


def _get_lock_file_path(lock_name):
    """Build a deterministic lock file path without exposing the raw lock name."""
    lock_digest = hashlib.sha256(lock_name.encode("utf-8")).hexdigest()
    if not os.path.exists(LOCK_FILE_DIR):
        os.makedirs(LOCK_FILE_DIR, mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR, exist_ok=True)
    return os.path.join(LOCK_FILE_DIR, f"{lock_digest}.lock")


def detect_all_meta_files(meta_files, local_path):
    """
    Checking that all meta files exit in local.

    Args:
        meta_files (List[str]): Names of meta files.
        local_path (str): Local path of dataset files.
    """

    all_meta_files = True
    for f in meta_files:
        dataset_file = os.path.basename(f)
        meta_file = dataset_file + ".db"
        if _detect_file_exist(local_path, meta_file, lock_file=meta_file) is False:
            all_meta_files = False
            break
    return all_meta_files


def make_sampler(shuffle, is_full_dataset, start, end):
    """
    Generate a proper sampler based on inputs.

    Args:
        Shuffle (Union[bool, Shuffle]): Shuffle level.
        is_full_dataset (bool): Whether to include full dataset file.
        start (int): Start index of sample for non-full dataset file.
        end (int): End index of sample for non-full dataset file.
    """

    sampler = None
    if shuffle in (Shuffle.GLOBAL, Shuffle.INFILE):
        if is_full_dataset:
            sampler = RandomSampler()
        else:
            sampler = SubsetRandomSampler(list(range(start, end)))
    else:
        if is_full_dataset:
            sampler = SequentialSampler()
        else:
            sampler = SubsetSampler(list(range(start, end)))
    return sampler


def make_shard_samples(dataset_file_size_list, size_per_shard, shard_id):
    """
    Make sharding files when shard_equal_rows equal to True.

    Args:
        dataset_file_size_list (List[tuple]): List of dataset file name and size.
        size_per_shard (int): Size of each sharding.
        shard_id (int): ID of sharding.
    """

    pre_cnt = 0
    shard_files = []
    finish = False
    while finish is False:
        for f, dataset_size in dataset_file_size_list:
            start_idx = shard_id * size_per_shard
            end_idx = (shard_id + 1) * size_per_shard
            start = 0
            end = dataset_size
            push = False
            is_full_dataset = False

            if pre_cnt <= start_idx < pre_cnt + dataset_size:
                start = start_idx - pre_cnt
                push = True
                if pre_cnt < end_idx <= pre_cnt + dataset_size:
                    end = end_idx - pre_cnt
                else:
                    end = dataset_size

            if start_idx <= pre_cnt < end_idx:
                start = 0
                push = True
                if pre_cnt + dataset_size >= end_idx:
                    end = end_idx - pre_cnt
                else:
                    end = dataset_size

            if push:
                if start == 0 and end == dataset_size:
                    is_full_dataset = True
                shard_files.append((f, start, end, is_full_dataset))
            pre_cnt += dataset_size
        if pre_cnt >= (shard_id + 1) * size_per_shard:
            finish = True
    return shard_files


def make_dataset_tuple(dataset_files, local_path):
    """
    Calculates the total size of the dataset and the size of each dataset file.

    Args:
        dataset_files (List[str]): Full paths of dataset files.
        local_path (str): Local directory path of dataset files.
    """

    dataset_file_size_list = []
    dataset_size = 0

    for dataset_file in dataset_files:
        meta_file = os.path.basename(dataset_file) + ".db"
        path = os.path.join(local_path, meta_file)
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM INDEXES")
                for row in cursor:
                    dataset_size += row[0]
                    dataset_file_size_list.append((dataset_file, row[0]))
        except Exception as e:
            raise RuntimeError("Failed to get dataset size from metadata, err: " + str(e)) from e
    return dataset_size, dataset_file_size_list


def make_shard_files(dataset_files, num_shards, shard_id):
    """
    Make sharding files when shard_equal_rows equal to False.

    Args:
        dataset_files (List[str]): Names of dataset files.
        num_shards (int): Number of all sharding.
        sharding (int): ID of sharding.
    """

    idx = 0
    shard_files = []

    for dataset_file in dataset_files:
        if idx % num_shards == shard_id:
            shard_files.append((dataset_file, -1, -1, True))
        idx += 1
    return shard_files


def get_bucket_and_key(obs_path):
    r"""
    Split OBS path to bucket name and object key.

    Args:
        obs_path (str): OBS path that starts with s3://.

    Returns:
        bucketName and objectKey.
    """

    start = obs_path.find("//")
    end = obs_path.find("/", start + 2)
    if end == -1:
        return obs_path[start + 2 :], ""
    return obs_path[start + 2 : end], obs_path[end + 1 :]


@exclusive_lock
def init_cache_and_queue(cache, q, path, shard_file, idx, is_full_dataset, lock_file="tmp"):
    """
    Initialize cache and queue according to the status of local dataset files.

    Args:
        cache (Dict[str, tuple]): Dict that indicate the status of local dataset files.
        q (Queue): Queue that pass dataset file to be download to thread.
        path (str): Local path of dataset file.
        shard_file (str): Full path of dataset file.
        idx (int): Index of dataset file.
        is_full_dataset (bool): Whether to include full dataset file.
        lock_file (str): File name to lock.
    """

    dataset_file = os.path.basename(shard_file)
    if os.path.exists(path):  # found in local
        logger.info(f"[{sys._getframe()} FUNCTION] Push dataset file {dataset_file} to cache.")  # pylint: disable=W0212
        cache[dataset_file] = (idx, not is_full_dataset)
    else:
        logger.info(
            f"[{sys._getframe()} FUNCTION] Push dataset file {dataset_file} "  # pylint: disable=W0212
            "to downloading queue."
        )
        cache[dataset_file] = (-1, not is_full_dataset)
        q.put((idx, shard_file))


@exclusive_lock
def _detect_file_exist(local_path, meta_file, lock_file="tmp"):
    """
    Detect that local meta file exists or not.

    Args:
        local_path (str): Local directory path of meta file.
        meta_file (str): Name of meta file.
        lock_file (str): File name to lock.
    """
    if os.path.exists(os.path.join(local_path, meta_file)):
        return True
    return False
