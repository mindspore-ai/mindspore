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
"""Data operations"""
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C

from .finetune_eval_config import gpt2_net_cfg


def create_language_model_dataset(device_num=1, repeat_count=1, rank_id=0, do_shuffle=True, dataset_path=""):
    """create dataset like language model task"""
    type_cast_op = C.TypeCast(mstype.int32)
    ds = de.MindDataset(dataset_path,
                        columns_list=["input_ids", "input_mask", "label_ids"],
                        shuffle=do_shuffle,
                        num_shards=device_num,
                        shard_id=rank_id)
    print("batch_size: {}".format(gpt2_net_cfg.batch_size))

    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="label_ids")
    ds = ds.batch(gpt2_net_cfg.batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)

    print("dataset size: {}".format(ds.get_dataset_size()))
    print("repeat count: {}".format(ds.get_repeat_count()))
    print("output shape: {}".format(ds.output_shapes()))
    print("output type: {}".format(ds.output_types()))
    print("============== create dataset successful ===============")

    return ds


def create_cbt_dataset(device_num=1, repeat_count=1, rank_id=0, do_shuffle=False, dataset_path=""):
    """create dataset for cbt task"""
    type_cast_op = C.TypeCast(mstype.int32)
    ds = de.MindDataset(dataset_path,
                        columns_list=["input_ids", "input_mask", "input_length", "mc_labels"],
                        shuffle=do_shuffle,
                        num_shards=device_num,
                        shard_id=rank_id)
    print("batch_size: {}".format(gpt2_net_cfg.batch_size))

    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="input_length")
    ds = ds.map(operations=type_cast_op, input_columns="mc_labels")
    ds = ds.batch(gpt2_net_cfg.batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)

    print("dataset size: {}".format(ds.get_dataset_size()))
    print("repeat count: {}".format(ds.get_repeat_count()))
    print("output shape: {}".format(ds.output_shapes()))
    print("output type: {}".format(ds.output_types()))
    print("============== create CBT LM dataset successful ===============")

    return ds


def create_lambada_control_dataset(device_num=1, repeat_count=1, rank_id=0, do_shuffle=True, dataset_path=""):
    """create dataset for lambada task"""
    type_cast_op = C.TypeCast(mstype.int32)
    ds = de.MindDataset(dataset_path,
                        columns_list=["input_ids", "input_mask", "input_length"],
                        shuffle=do_shuffle,
                        num_shards=device_num,
                        shard_id=rank_id)
    print("batch_size: {}".format(gpt2_net_cfg.batch_size))

    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="input_length")
    ds = ds.batch(gpt2_net_cfg.batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)

    print("dataset size: {}".format(ds.get_dataset_size()))
    print("repeat count: {}".format(ds.get_repeat_count()))
    print("output shape: {}".format(ds.output_shapes()))
    print("output type: {}".format(ds.output_types()))
    print("============== create dataset successful ===============")
    return ds
