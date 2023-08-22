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
This dataset module provides internal utility function for OBSMindDataset API.
"""

import fcntl
import os
import shutil
import stat
import sys
import sqlite3
import time

from functools import wraps
from obs import ObsClient

from mindspore import log as logger
from .config_loader import config
from ..datasets import Shuffle
from ..samplers import RandomSampler, SequentialSampler, SubsetSampler, SubsetRandomSampler

obsClient = ObsClient(
    access_key_id=config.AK,
    secret_access_key=config.SK,
    server=config.SERVER
)


def get_used_disk_per():
    """ Get the disk usage of working directory."""

    if not os.path.exists(config.WORKING_PATH):
        try:
            os.makedirs(config.WORKING_PATH)
        except FileExistsError:
            pass

    total, used, _ = shutil.disk_usage(config.WORKING_PATH)
    return used / total


def try_load_from_obs(remote_path, dataset_file, local_path):
    """
    Download all dataset files from OBS, skip if it exists.

    Args:
        remote_path (str): OBS path of dataset files.
        dataset_file (str): Name of dataset file.
        local_path (str): Local path of dataset files.
    """

    if not os.path.exists(os.path.join(local_path, dataset_file)):
        _download_file(remote_path, dataset_file, local_path, lock_file=dataset_file)
    meta_file = dataset_file + '.db'
    if not os.path.exists(os.path.join(local_path, meta_file)):
        _download_file(remote_path, meta_file, local_path, lock_file=meta_file)


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
        meta_file = dataset_file + '.db'
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
        meta_file = os.path.basename(dataset_file) + '.db'
        path = os.path.join(local_path, meta_file)
        try:
            conn = sqlite3.connect(path)
            c = conn.cursor()
            cursor = c.execute("SELECT COUNT(*) FROM INDEXES")
            for row in cursor:
                dataset_size += row[0]
                dataset_file_size_list.append((dataset_file, row[0]))
            conn.close()
        except Exception as e:
            raise RuntimeError(
                "Failed to get dataset size from metadata, err: " + str(e))
    return dataset_size, dataset_file_size_list


def fetch_meta_files(meta_files, local_path):
    """
    Download all meta files from obs, skip if it exists.

    Args:
        meta_files (List[str]): Full paths of meta files.
        local_path (str): Local directory path of dataset files.
    """

    for df in meta_files:
        dataset_file = os.path.basename(df)
        meta_file = dataset_file + '.db'
        remote_path = os.path.dirname(df)
        _download_file(remote_path, meta_file, local_path, lock_file=meta_file)


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

    start = obs_path.find('//')
    end = obs_path.find('/', start + 2)
    if end == -1:
        return obs_path[start + 2:], ""
    return obs_path[start + 2:end], obs_path[end + 1:]


def exclusive_lock(func):
    """ Decorator that execute func under exclusive lock. """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            lock_file = os.path.join('/tmp/', '{}.lock'.format(kwargs['lock_file']))
        except KeyError:
            raise RuntimeError("Lock file can not found in function {}.".format(func_name))
        with open(lock_file, 'w') as fd:
            retry_cnt = 0
            success = False
            while True:
                if success:
                    break
                try:
                    if retry_cnt > config.MAX_RETRY:
                        raise RuntimeError("Function {} retries times {} has exceeded threshold {}.".format(
                            func_name, retry_cnt, config.MAX_RETRY))
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    success = True
                    result = func(*args, **kwargs)
                except RuntimeError as e:
                    raise e
                except Exception as e: # pylint: disable=W0703
                    retry_cnt += 1
                    import traceback
                    logger.error(traceback.format_exc())
                    time.sleep(config.RETRY_DELTA_TIME)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        if os.path.exists(lock_file):
            os.chmod(lock_file, stat.S_IRUSR | stat.S_IWUSR)
        return result
    return wrapped_func


def retry_execute(func):
    """ Decorator that retry on unexpected errors. """

    func_name = func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        retry_cnt = 0
        success = False
        while True:
            if success:
                break
            try:
                if retry_cnt >= config.MAX_RETRY:
                    err_msg = "Function {} has retried for {} times, please check error above.".format(
                        func_name, retry_cnt)
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)
                result = func(*args, **kwargs)
                success = True
            except RuntimeError as e:
                raise e
            except Exception: # pylint: disable=W0703
                retry_cnt += 1
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(config.RETRY_DELTA_TIME)
        return result
    return wrapper


@retry_execute
def _check_file_exists_in_obs(obs_path):
    """
    Detect that file exists in OBS.

    Args:
        obs_path (str): OBS path of dataset file.
    """

    bucket_name, object_key = get_bucket_and_key(obs_path)
    try:
        resp = obsClient.getObjectMetadata(bucket_name, object_key)
    except ConnectionRefusedError:
        err_msg = "Failed to connect to OBS, please check OBS sever {}:{}.".format(obsClient.server, obsClient.port)
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    if resp.status < 300:
        logger.debug("[{} FUNCTION] OBS requestId: {}.".format(
            sys._getframe(), resp.requestId)) # pylint: disable=W0212
    elif resp.status == 403:
        err_msg = "OBS access is Forbidden, please check AK or SK."
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    else:
        err_msg = "File {} not found in OBS, please check again.".format(obs_path)
        logger.error(err_msg)
        raise RuntimeError(err_msg)


@retry_execute
def _file_download_from_obs(obs_path, local_path):
    """
    Download file from OBS.

    Args:
        obs_path (str): OBS path of dataset file.
        local_path (str): Local path of dataset file.
    """

    bucket_name, object_key = get_bucket_and_key(obs_path)
    downloadFile = local_path
    taskNum = config.TASK_NUM
    partSize = config.PART_SIZE
    enableCheckpoint = True

    resp = obsClient.downloadFile(
        bucket_name, object_key, downloadFile, partSize, taskNum, enableCheckpoint)
    if resp.status < 300:
        logger.debug("[{} FUNCTION] OBS requestId: {}.".format(
            sys._getframe(), resp.requestId)) # pylint: disable=W0212
    else:
        raise Exception("OBS SDK errorCode:{}, errMsg: {}.".format(
            resp.errorCode, resp.errorMessage))


@exclusive_lock
def _download_file(remote_path, object_name, des_path, lock_file='tmp'):
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

    _check_file_exists_in_obs(obs_path)
    _file_download_from_obs(obs_path, local_path)


@exclusive_lock
def init_cache_and_queue(cache, q, path, shard_file, idx, is_full_dataset, lock_file='tmp'):
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
        logger.info("[{} FUNCTION] Push dataset file {} to cache.".format(
            sys._getframe(), dataset_file)) # pylint: disable=W0212
        cache[dataset_file] = (idx, not is_full_dataset)
    else:
        logger.info("[{} FUNCTION] Push dataset file {} to downloading queue.".format(
            sys._getframe(), dataset_file)) # pylint: disable=W0212
        cache[dataset_file] = (-1, not is_full_dataset)
        q.put((idx, shard_file))


@exclusive_lock
def _detect_file_exist(local_path, meta_file, lock_file='tmp'):
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


@retry_execute
def file_upload_to_obs(obs_path, sync_dir, ready_file_name):
    """
    Upload sync file to OBS.

    Args:
        obs_path (str): OBS path of dataset file.
        sync_fir (str): OBS directory path used for synchronization.
        ready_file_name (str): Name of synchronization file.
    """

    bucket_name, object_key = get_bucket_and_key(obs_path)

    if not object_key:
        resp = obsClient.headBucket(bucket_name)
    else:
        if not object_key.endswith("/"):
            object_key += "/"
        resp = obsClient.getObjectMetadata(bucket_name, object_key)

    if resp.status < 300:
        logger.debug("[{} FUNCTION] OBS requestId: {}.".format(
            sys._getframe(), resp.requestId)) # pylint: disable=W0212
    else:
        raise RuntimeError("Directory/Bucket used for synchronization {} is not found in OBS, " \
                           "please create it on OBS first.".format(obs_path))

    remote_dir = os.path.join(object_key, sync_dir)
    resp = obsClient.putContent(bucket_name, remote_dir, content=None)
    if resp.status < 300:
        logger.debug("[{} FUNCTION] OBS requestId: {}.".format(
            sys._getframe(), resp.requestId)) # pylint: disable=W0212
    else:
        raise Exception("OBS SDK errorCode:{}, errMsg: {}.".format(
            resp.errorCode, resp.errorMessage))
    resp = obsClient.putContent(bucket_name, os.path.join(
        remote_dir, ready_file_name), content='OK')
    if resp.status < 300:
        logger.debug("[{} FUNCTION] OBS requestId: {}.".format(
            sys._getframe(), resp.requestId)) # pylint: disable=W0212
    else:
        raise Exception("OBS SDK errorCode:{}, errMsg: {}.".format(
            resp.errorCode, resp.errorMessage))
