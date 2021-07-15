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
"""
Data operations, will be used in run_pretrain.py
"""

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger
from .config import cfg

def create_ernie_dataset(device_num=1, rank=0, do_shuffle=True, data_file_path=None, schema_dir=None):
    """create train dataset"""
    # apply repeat operations
    data_set = ds.MindDataset(data_file_path,
                              columns_list=["input_ids", "input_mask", "token_type_id", "next_sentence_labels",
                                            "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                              shuffle=do_shuffle,
                              num_shards=device_num,
                              shard_id=rank)

    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="next_sentence_labels")
    data_set = data_set.map(operations=type_cast_op, input_columns="token_type_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    data_set = data_set.batch(cfg.batch_size, drop_remainder=True)
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set

def create_finetune_dataset(batch_size=1,
                            repeat_count=1,
                            data_file_path=None,
                            rank_size=1,
                            rank_id=0,
                            do_shuffle=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.MindDataset(data_file_path,
                              columns_list=["input_ids", "input_mask", "token_type_id", "label_ids"],
                              shuffle=do_shuffle,
                              num_shards=rank_size,
                              shard_id=rank_id)
    data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="token_type_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.repeat(repeat_count)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set

def create_mrc_dataset(batch_size=1,
                       repeat_count=1,
                       data_file_path=None,
                       rank_size=1,
                       rank_id=0,
                       do_shuffle=True,
                       is_training=True,
                       drop_reminder=False):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if is_training:
        data_set = ds.MindDataset(data_file_path,
                                  columns_list=["input_ids", "input_mask", "token_type_id",
                                                "start_position", "end_position", "unique_id"],
                                  shuffle=do_shuffle,
                                  num_shards=rank_size,
                                  shard_id=rank_id)
        data_set = data_set.map(operations=type_cast_op, input_columns="token_type_id")
        data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
        data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
        data_set = data_set.map(operations=type_cast_op, input_columns="start_position")
        data_set = data_set.map(operations=type_cast_op, input_columns="end_position")
        data_set = data_set.map(operations=type_cast_op, input_columns="unique_id")
    else:
        data_set = ds.MindDataset(data_file_path,
                                  columns_list=["input_ids", "input_mask", "token_type_id", "unique_id"],
                                  shuffle=do_shuffle,
                                  num_shards=rank_size,
                                  shard_id=rank_id)
        data_set = data_set.map(operations=type_cast_op, input_columns="token_type_id")
        data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
        data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
        data_set = data_set.map(operations=type_cast_op, input_columns="unique_id")
    data_set = data_set.repeat(repeat_count)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=drop_reminder)
    return data_set
