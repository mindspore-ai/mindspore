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
"""Dataset loader to feed into model."""
import mindspore.dataset as ds


def load_dataset(input_files, batch_size, sink_mode=False,
                 rank_size=1, rank_id=0):
    """
    Load dataset according to passed in params.

    Args:
        input_files (list): Data files.
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
        raise ValueError("Require at least one dataset.")

    if not isinstance(sink_mode, bool):
        raise ValueError("`sink` must be type of bool.")

    for datafile in input_files:
        print(" | Loading", datafile, ".")

    data_set = ds.MindDataset(
        input_files, columns_list=["content", "sen_len", "aspect", "solution"],
        shuffle=False, num_shards=rank_size, shard_id=rank_id,
        num_parallel_workers=8)

    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())

    ori_dataset_size = data_set.get_dataset_size()
    print(" | Dataset size", ori_dataset_size, ".")

    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
