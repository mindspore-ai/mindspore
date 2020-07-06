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
"""Dataset loader to feed into model."""
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as deC


def _load_dataset(input_files, batch_size, epoch_count=1,
                  sink_mode=False, sink_step=1, rank_size=1, rank_id=0, shuffle=True):
    """
    Load dataset according to passed in params.

    Args:
        input_files (list): Data files.
        batch_size (int): Batch size.
        epoch_count (int): Epoch count.
        sink_mode (bool): Whether enable sink mode.
        sink_step (int): Step to sink.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.

    Returns:
        Dataset, dataset instance.
    """
    if not input_files:
        raise FileNotFoundError("Require at least one dataset.")

    if not isinstance(sink_mode, bool):
        raise ValueError("`sink` must be type of bool.")

    for datafile in input_files:
        print(f" | Loading {datafile}.")

    ds = de.TFRecordDataset(
        input_files,
        columns_list=[
            "src", "src_padding",
            "prev_opt", "prev_padding",
            "target", "tgt_padding"
        ],
        shuffle=shuffle, num_shards=rank_size, shard_id=rank_id,
        shard_equal_rows=True, num_parallel_workers=8)

    ori_dataset_size = ds.get_dataset_size()
    print(f" | Dataset size: {ori_dataset_size}.")
    repeat_count = epoch_count
    if sink_mode:
        ds.set_dataset_size(sink_step * batch_size)
        repeat_count = epoch_count * ori_dataset_size // ds.get_dataset_size()

    type_cast_op = deC.TypeCast(mstype.int32)
    ds = ds.map(input_columns="src", operations=type_cast_op)
    ds = ds.map(input_columns="src_padding", operations=type_cast_op)
    ds = ds.map(input_columns="prev_opt", operations=type_cast_op)
    ds = ds.map(input_columns="prev_padding", operations=type_cast_op)
    ds = ds.map(input_columns="target", operations=type_cast_op)
    ds = ds.map(input_columns="tgt_padding", operations=type_cast_op)

    ds = ds.rename(
        input_columns=["src",
                       "src_padding",
                       "prev_opt",
                       "prev_padding",
                       "target",
                       "tgt_padding"],
        output_columns=["source_eos_ids",
                        "source_eos_mask",
                        "target_sos_ids",
                        "target_sos_mask",
                        "target_eos_ids",
                        "target_eos_mask"]
    )

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)

    ds.channel_name = 'transformer'
    return ds


def load_dataset(data_files: list, batch_size: int, epoch_count: int,
                 sink_mode: bool, sink_step: int = 1, rank_size: int = 1, rank_id: int = 0, shuffle=True):
    """
    Load dataset.

    Args:
        data_files (list): Data files.
        batch_size (int): Batch size.
        epoch_count (int): Epoch count.
        sink_mode (bool): Whether enable sink mode.
        sink_step (int): Step to sink.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.

    Returns:
        Dataset, dataset instance.
    """
    return _load_dataset(data_files, batch_size, epoch_count, sink_mode,
                         sink_step, rank_size, rank_id, shuffle=shuffle)
