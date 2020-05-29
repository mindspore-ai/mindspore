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
"""Data operations, will be used in train.py."""

import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as deC
from mindspore import log as logger
from .config import transformer_net_cfg

def create_transformer_dataset(epoch_count=1, rank_size=1, rank_id=0, do_shuffle="true", enable_data_sink="true",
                               dataset_path=None):
    """create dataset"""
    repeat_count = epoch_count
    ds = de.MindDataset(dataset_path,
                        columns_list=["source_eos_ids", "source_eos_mask",
                                      "target_sos_ids", "target_sos_mask",
                                      "target_eos_ids", "target_eos_mask"],
                        shuffle=(do_shuffle == "true"), num_shards=rank_size, shard_id=rank_id)

    type_cast_op = deC.TypeCast(mstype.int32)
    ds = ds.map(input_columns="source_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="source_eos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_mask", operations=type_cast_op)

    # apply batch operations
    ds = ds.batch(transformer_net_cfg.batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)

    ds.channel_name = 'transformer'
    logger.info("data size: {}".format(ds.get_dataset_size()))
    logger.info("repeatcount: {}".format(ds.get_repeat_count()))
    return ds, repeat_count
