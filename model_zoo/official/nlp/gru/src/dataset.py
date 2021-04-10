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
"""Data operations, will be used in train.py."""

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as deC
from src.config import config

de.config.set_seed(1)

def random_teacher_force(source_ids, target_ids, target_mask):

    teacher_force = np.random.random() < config.teacher_force_ratio
    teacher_force_array = np.array([teacher_force], dtype=bool)
    return source_ids, target_ids, teacher_force_array

def create_gru_dataset(epoch_count=1, batch_size=1, rank_size=1, rank_id=0, do_shuffle=True, dataset_path=None,
                       is_training=True):
    """create dataset"""
    ds = de.MindDataset(dataset_path,
                        columns_list=["source_ids", "target_ids",
                                      "target_mask"],
                        shuffle=do_shuffle, num_parallel_workers=10, num_shards=rank_size, shard_id=rank_id)
    operations = random_teacher_force
    ds = ds.map(operations=operations, input_columns=["source_ids", "target_ids", "target_mask"],
                output_columns=["source_ids", "target_ids", "teacher_force"],
                column_order=["source_ids", "target_ids", "teacher_force"])
    type_cast_op = deC.TypeCast(mstype.int32)
    type_cast_op_bool = deC.TypeCast(mstype.bool_)
    ds = ds.map(operations=type_cast_op, input_columns="source_ids")
    ds = ds.map(operations=type_cast_op, input_columns="target_ids")
    ds = ds.map(operations=type_cast_op_bool, input_columns="teacher_force")
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(1)
    return ds
