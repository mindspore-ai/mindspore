# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
from mindspore import log as logger
from .config import bert_net_cfg


def create_bert_dataset(epoch_size=1, device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None):
    """create train dataset"""
    # apply repeat operations
    repeat_count = epoch_size
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if "tfrecord" in file_name:
            data_files.append(os.path.join(data_dir, file_name))
    data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                  columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                  shuffle=(do_shuffle == "true"), num_shards=device_num, shard_id=rank,
                                  shard_equal_rows=True)
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    new_repeat_count = int(repeat_count * ori_dataset_size // data_set.get_dataset_size())
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="next_sentence_labels")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    data_set = data_set.batch(bert_net_cfg.batch_size, drop_remainder=True)
    data_set = data_set.repeat(max(new_repeat_count, repeat_count))
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeatcount: {}".format(data_set.get_repeat_count()))
    return data_set, new_repeat_count
