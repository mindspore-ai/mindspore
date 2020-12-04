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
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as deC


def _load_dataset(input_files, schema_file, batch_size, sink_mode=False,
                  rank_size=1, rank_id=0, shuffle=True, drop_remainder=True,
                  is_translate=False):
    """
    Load dataset according to passed in params.

    Args:
        input_files (list): Data files.
        schema_file (str): Schema file path.
        batch_size (int): Batch size.
        sink_mode (bool): Whether enable sink mode.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.
        drop_remainder (bool): Whether drop the last possibly incomplete batch.
        is_translate (bool): Whether translate the text.

    Returns:
        Dataset, dataset instance.
    """
    if not input_files:
        raise FileNotFoundError("Require at least one dataset.")

    if not (schema_file and
            os.path.exists(schema_file)
            and os.path.isfile(schema_file)
            and os.path.basename(schema_file).endswith(".json")):
        raise FileNotFoundError("`dataset_schema` must be a existed json file.")

    if not isinstance(sink_mode, bool):
        raise ValueError("`sink` must be type of bool.")

    for datafile in input_files:
        print(f" | Loading {datafile}.")

    if not is_translate:
        ds = de.MindDataset(
            input_files, columns_list=[
                "src", "src_padding",
                "prev_opt",
                "target", "tgt_padding"
            ], shuffle=False, num_shards=rank_size, shard_id=rank_id,
            num_parallel_workers=8
        )

        ori_dataset_size = ds.get_dataset_size()
        print(f" | Dataset size: {ori_dataset_size}.")
        if shuffle:
            ds = ds.shuffle(buffer_size=ori_dataset_size // 20)
        type_cast_op = deC.TypeCast(mstype.int32)
        ds = ds.map(input_columns="src", operations=type_cast_op, num_parallel_workers=8)
        ds = ds.map(input_columns="src_padding", operations=type_cast_op, num_parallel_workers=8)
        ds = ds.map(input_columns="prev_opt", operations=type_cast_op, num_parallel_workers=8)
        ds = ds.map(input_columns="target", operations=type_cast_op, num_parallel_workers=8)
        ds = ds.map(input_columns="tgt_padding", operations=type_cast_op, num_parallel_workers=8)

        ds = ds.rename(
            input_columns=["src",
                           "src_padding",
                           "prev_opt",
                           "target",
                           "tgt_padding"],
            output_columns=["source_eos_ids",
                            "source_eos_mask",
                            "target_sos_ids",
                            "target_eos_ids",
                            "target_eos_mask"]
        )
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    else:
        ds = de.MindDataset(
            input_files, columns_list=[
                "src", "src_padding"
            ],
            shuffle=False, num_shards=rank_size, shard_id=rank_id,
            num_parallel_workers=8
        )

        ori_dataset_size = ds.get_dataset_size()
        print(f" | Dataset size: {ori_dataset_size}.")
        if shuffle:
            ds = ds.shuffle(buffer_size=ori_dataset_size // 20)
        type_cast_op = deC.TypeCast(mstype.int32)
        ds = ds.map(input_columns="src", operations=type_cast_op, num_parallel_workers=8)
        ds = ds.map(input_columns="src_padding", operations=type_cast_op, num_parallel_workers=8)

        ds = ds.rename(
            input_columns=["src",
                           "src_padding"],
            output_columns=["source_eos_ids",
                            "source_eos_mask"]
        )
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    return ds


def load_dataset(data_files: list, schema: str, batch_size: int, sink_mode: bool,
                 rank_size: int = 1, rank_id: int = 0, shuffle=True, drop_remainder=True, is_translate=False):
    """
    Load dataset.

    Args:
        data_files (list): Data files.
        schema (str): Schema file path.
        batch_size (int): Batch size.
        sink_mode (bool): Whether enable sink mode.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.

    Returns:
        Dataset, dataset instance.
    """
    return _load_dataset(data_files, schema, batch_size, sink_mode, rank_size, rank_id, shuffle=shuffle,
                         drop_remainder=drop_remainder, is_translate=is_translate)
