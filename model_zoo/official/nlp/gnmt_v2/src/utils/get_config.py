# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Get Config."""
import os
from typing import List
import mindspore.common.dtype as mstype

def _is_dataset_file(file: str):
    return "tfrecord" in file.lower() or "mindrecord" in file.lower()

def _get_files_from_dir(folder: str):
    _files = []
    for file in os.listdir(folder):
        if _is_dataset_file(file):
            _files.append(os.path.join(folder, file))
    return _files

def get_source_list(folder: str) -> List:
    """
    Get file list from a folder.

    Returns:
        list, file list.
    """
    _list = []
    if not folder:
        return _list

    if os.path.isdir(folder):
        _list = _get_files_from_dir(folder)
    else:
        if _is_dataset_file(folder):
            _list.append(folder)
    return _list

def get_config(config):
    '''get config.'''
    config.pre_train_dataset = None if config.pre_train_dataset == "" else config.pre_train_dataset
    config.fine_tune_dataset = None if config.fine_tune_dataset == "" else config.fine_tune_dataset
    config.valid_dataset = None if config.valid_dataset == "" else config.valid_dataset
    config.test_dataset = None if config.test_dataset == "" else config.test_dataset
    if hasattr(config, 'test_tgt'):
        config.test_tgt = None if config.test_tgt == "" else config.test_tgt

    config.pre_train_dataset = get_source_list(config.pre_train_dataset)
    config.fine_tune_dataset = get_source_list(config.fine_tune_dataset)
    config.valid_dataset = get_source_list(config.valid_dataset)
    config.test_dataset = get_source_list(config.test_dataset)

    if not isinstance(config.epochs, int) and config.epochs < 0:
        raise ValueError("`epoch` must be type of int.")

    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config
