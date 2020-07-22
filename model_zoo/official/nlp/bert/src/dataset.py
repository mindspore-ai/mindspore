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
"""
Data operations, will be used in run_pretrain.py
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger
from .config import bert_net_cfg


def create_bert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None):
    """create train dataset"""
    # apply repeat operations
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if "tfrecord" in file_name:
            data_files.append(os.path.join(data_dir, file_name))
    ds = de.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                            columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                          "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                            shuffle=de.Shuffle.FILES if do_shuffle == "true" else False,
                            num_shards=device_num, shard_id=rank, shard_equal_rows=True)
    ori_dataset_size = ds.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    ds = ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    ds = ds.map(input_columns="next_sentence_labels", operations=type_cast_op)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    # apply batch operations
    ds = ds.batch(bert_net_cfg.batch_size, drop_remainder=True)
    logger.info("data size: {}".format(ds.get_dataset_size()))
    logger.info("repeat count: {}".format(ds.get_repeat_count()))
    return ds


def create_ner_dataset(batch_size=1, repeat_count=1, assessment_method="accuracy",
                       data_file_path=None, schema_file_path=None):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    ds = de.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                            columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"])
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        ds = ds.map(input_columns="label_ids", operations=type_cast_op_float)
    else:
        ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)
    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def create_classification_dataset(batch_size=1, repeat_count=1, assessment_method="accuracy",
                                  data_file_path=None, schema_file_path=None):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    ds = de.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                            columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"])
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        ds = ds.map(input_columns="label_ids", operations=type_cast_op_float)
    else:
        ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)
    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def create_squad_dataset(batch_size=1, repeat_count=1, data_file_path=None, schema_file_path=None, is_training=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if is_training:
        ds = de.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                columns_list=["input_ids", "input_mask", "segment_ids",
                                              "start_positions", "end_positions",
                                              "unique_ids", "is_impossible"])
        ds = ds.map(input_columns="start_positions", operations=type_cast_op)
        ds = ds.map(input_columns="end_positions", operations=type_cast_op)
    else:
        ds = de.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                columns_list=["input_ids", "input_mask", "segment_ids", "unique_ids"])
        ds = ds.map(input_columns="input_ids", operations=type_cast_op)
        ds = ds.map(input_columns="input_mask", operations=type_cast_op)
        ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)
    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
