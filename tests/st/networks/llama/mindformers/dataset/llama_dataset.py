# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Causal Image Modeling Dataset."""
import copy
import os
import re
from typing import Union, Optional, Callable
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset import MindDataset
from mindspore.dataset.transforms import TypeCast

from mindformers.llama_utils import get_dataset_map, get_real_group_size, get_real_rank
from mindformers.normal_config import MindFormerConfig

def get_input_data_batch_slice_map(input_ids, eod_token_id, dis, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Args:
        input_ids: the input token ids
        eod_token_id: the id for <EOD>
        dis: the slice value for each rank
        rank_id: the current rank id
    Returns:
        batch_input_ids: the input token ids
        batch_position_ids: the position ids cosidering eod reset
        batch_attention_mask: the attention mask considering eod reset
    """
    rank = int(rank_id)
    input_ids = input_ids[rank * dis: (rank + 1) * dis]
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))

    # Loop through batches
    for bs_i in range(len(input_ids)):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find the index of <EOS>
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_token_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering <EOS>
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


class BaseDataset:
    """
    Base Dataset.

    Args:
        dataset_config (dict): Config for dataset.

    """

    def __init__(self, dataset_config: dict = None):
        self.dataset_config = dataset_config

    @classmethod
    def check_dataset_config(cls, dataset_config, params):
        """Check `dataset_config`, If it is empty, use the input parameter to create a new `dataset_config`."""
        if not dataset_config:
            params.pop("dataset_config")
            kwargs = params.pop("kwargs") if params.get("kwargs") else {}
            params.update(kwargs)
            dataset_config = MindFormerConfig(**params)
        return dataset_config

    @classmethod
    def init_dataset_config(cls, dataset_config):
        """Init dataset config."""
        ds.config.set_seed(dataset_config.seed)
        ds.config.set_prefetch_size(dataset_config.prefetch_size)
        ds.config.set_numa_enable(dataset_config.numa_enable)

        if dataset_config.auto_tune:
            if dataset_config.profile:
                raise EnvironmentError(
                    "MindSpore's AutoTune is enabled, so Profile cannot be enabled,"
                    "now Profile's flag is True, please set to False!")
            os.makedirs(dataset_config.filepath_prefix, exist_ok=True)
            dataset_config.filepath_prefix = os.path.join(dataset_config.filepath_prefix, "autotune")
            ds.config.set_enable_autotune(True, filepath_prefix=dataset_config.filepath_prefix)
            ds.config.set_autotune_interval(dataset_config.autotune_per_step)

    @classmethod
    def _generate_shard_info(cls):
        """Generate shard info for dataset"""
        rank_id = get_real_rank()
        device_num = get_real_group_size()
        return cls._check_device_rank_for_parallel(rank_id, device_num)

    @classmethod
    def _check_device_rank_for_parallel(cls, rank_id, device_num):
        """Check device num and rank id in auto parallel mode."""
        if cls._is_semi_full_batch():
            rank_id = None
            device_num = None
        return rank_id, device_num

    @classmethod
    def _is_semi_full_batch(cls):
        return ((ms.context.get_auto_parallel_context("parallel_mode") in ['semi_auto_parallel', 'auto_parallel'])
                and ms.context.get_auto_parallel_context("full_batch"))

    @classmethod
    def _is_data_parallel(cls):
        return ms.context.get_auto_parallel_context("parallel_mode") == ms.context.ParallelMode.DATA_PARALLEL


class CausalLanguageModelDataset(BaseDataset):
    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                eod_reset: bool = False,
                eod_token_id: Optional[int] = None,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num

        dataset = cls._process_mindrecord_data(dataset_config)

        type_cast_op = TypeCast(mstype.int32)
        if dataset_config.eod_reset:
            if cls._is_semi_full_batch() or cls._is_data_parallel():
                rank_id = 0
                dis = dataset_config.batch_size
            else:
                # Each card slice a small batch from the full batch
                dis = dataset_config.batch_size // device_num
                if dataset_config.batch_size % device_num != 0:
                    raise ValueError(
                        f"batch size {dataset_config.batch_size} should be a multiple of device number {device_num}."
                        " You should change the args: per_batch_size.")

            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns)
            map_func = lambda input_ids: get_input_data_batch_slice_map(input_ids,
                                                                        eod_token_id=dataset_config.eod_token_id,
                                                                        rank_id=rank_id,
                                                                        dis=dis)
            dataset = get_dataset_map(dataset, map_func,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns)
            dataset = dataset.project(columns=dataset_config.output_columns)

            for input_arg in dataset_config.output_columns:
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        else:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns,
                                    num_parallel_workers=dataset_config.num_parallel_workers)
            dataset = dataset.project(columns=dataset_config.input_columns)
            for input_arg in dataset_config.input_columns:
                dataset = get_dataset_map(dataset, type_cast_op,
                                          input_columns=input_arg)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        dataset_files = []
        mind_compile = re.compile("mindrecord0*$")
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if re.findall(mind_compile, file) or file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if re.findall(mind_compile, data_dir) or data_dir.endswith(".tfrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        data_loader_dict = {
            "shuffle": dataset_config.data_loader.shuffle,
            "dataset_files": dataset_files,
            "num_shards": dataset_config.device_num,
            "shard_id": dataset_config.rank_id,
            "columns_list": dataset_config.input_columns
        }
        dataset = MindDataset(**data_loader_dict)
        return dataset
