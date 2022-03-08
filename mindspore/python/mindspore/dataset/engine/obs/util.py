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


def try_load_from_obs(remote_path, dataset_file, local_path, shard_id):
    """ Download all dataset files from obs, skip if it exists. """
    try:
        if not os.path.exists(os.path.join(local_path, dataset_file)):
            download_file(remote_path, dataset_file, local_path, lock_file=dataset_file)
        meta_file = dataset_file + '.db'
        if not os.path.exists(os.path.join(local_path, meta_file)):
            download_file(remote_path, meta_file, local_path, lock_file=meta_file)
    except Exception as e:
        raise RuntimeError("Failed to fetch file from obs, error: " + str(e))


def detect_all_meta_files(dataset_files, local_path):
    """ Checking that all meta files exit on local. """

    all_meta_files = True
    for f in dataset_files:
        dataset_file = os.path.basename(f)
        meta_file = dataset_file + '.db'
        if detect_file_exist(local_path, meta_file, lock_file=meta_file) is False:
            all_meta_files = False
            break
    return all_meta_files


def make_sampler(shuffle, is_full_dataset, start, end):
    """ Generate a proper sampler based on inputs. """

    sampler = None
    if shuffle == Shuffle.GLOBAL:
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
    """ Make sharding files when shard_equal_rows is True. """
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
    """ Calculates the total size of the dataset and the size of each dataset file """

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


def fetch_meta_files(dataset_files, local_path, shard_id):
    """ Download all meta files from obs, skip if it exists"""

    try:
        for df in dataset_files:
            dataset_file = os.path.basename(df)
            meta_file = dataset_file + '.db'
            remote_path = os.path.dirname(df)
            download_file(remote_path, meta_file, local_path, lock_file=meta_file)
    except Exception as e:
        raise RuntimeError(
            "Failed to fetch meta file from OBS, error: " + str(e))


def make_shard_files(dataset_files, num_shards, shard_id):
    """ Make sharding files when shard_equal_rows is False. """

    idx = 0
    shard_files = []

    for dataset_file in dataset_files:
        if idx % num_shards == shard_id:
            shard_files.append((dataset_file, -1, -1, True))
        idx += 1
    return shard_files


def get_bucket_and_key(obs_path):
    r"""
    split obs path to bucket name and object key.

    Args:
        obs_path: obs path that starts with s3://.

    Returns:
        (str, str), bucketName and objectKey.
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
                if retry_cnt > config.MAX_RETRY:
                    err_msg = "Function {} retries times {} has exceeded threshold {}.".format(
                        func_name, retry_cnt, config.MAX_RETRY)
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
def check_file_exists_in_obs(obs_path):
    """ Detect that file exists in obs. """

    bucket_name, object_key = get_bucket_and_key(obs_path)
    resp = obsClient.getObjectMetadata(bucket_name, object_key)
    if resp.status < 300:
        logger.debug("[{} FUNCTION] OBS requestId: {}.".format(
            sys._getframe(), resp.requestId)) # pylint: disable=W0212
    else:
        err_msg = "File {} not found in OBS, please check again.".format(obs_path)
        logger.error(err_msg)
        raise RuntimeError(err_msg)


@retry_execute
def file_download_from_obs(obs_path, local_path):
    """ Download file from OBS. """

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
def download_file(remote_path, object_name, des_path, lock_file='tmp'):
    """ Download file from OBS exclusively. """

    local_path = os.path.join(des_path, object_name)
    if os.path.exists(local_path):
        return
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    obs_path = os.path.join(remote_path, object_name)

    check_file_exists_in_obs(obs_path)
    file_download_from_obs(obs_path, local_path)


@exclusive_lock
def init_cache_and_queue(cache, q, path, shard_file, idx, is_full_dataset, lock_file='tmp'):
    """ Initialize cache and queue according to the status of local dataset files."""

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
def detect_file_exist(local_path, meta_file, lock_file='tmp'):
    """ Detect that local dataset file exists or not. """
    if os.path.exists(os.path.join(local_path, meta_file)):
        return True
    return False


@retry_execute
def file_upload_to_obs(obs_path, sync_dir, ready_file_name):
    """ Upload sync file to OBS. """

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
