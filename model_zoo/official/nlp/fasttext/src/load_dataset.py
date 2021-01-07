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
"""FastText data loader"""
import mindspore.dataset as ds


def load_dataset(dataset_path,
                 batch_size,
                 epoch_count=1,
                 rank_size=1,
                 rank_id=0,
                 bucket=None,
                 shuffle=True):
    """dataset loader"""

    def batch_per_bucket(bucket_length, input_file):
        input_file = input_file + '/train_dataset_bs_' + str(bucket_length) + '.mindrecord'
        if not input_file:
            raise FileNotFoundError("input file parameter must not be empty.")

        data_set = ds.MindDataset(input_file,
                                  columns_list=['src_tokens', 'src_tokens_length', 'label_idx'],
                                  shuffle=shuffle,
                                  num_shards=rank_size,
                                  shard_id=rank_id,
                                  num_parallel_workers=4)
        ori_dataset_size = data_set.get_dataset_size()
        print(f"Dataset size: {ori_dataset_size}")
        repeat_count = epoch_count

        data_set = data_set.rename(input_columns=['src_tokens', 'src_tokens_length', 'label_idx'],
                                   output_columns=['src_token_text', 'src_tokens_text_length', 'label_idx_tag'])
        data_set = data_set.batch(batch_size, drop_remainder=False)
        data_set = data_set.repeat(repeat_count)
        return data_set

    for i, _ in enumerate(bucket):
        bucket_len = bucket[i]
        ds_per = batch_per_bucket(bucket_len, dataset_path)
        if i == 0:
            data_set = ds_per
        else:
            data_set = data_set + ds_per
    data_set = data_set.shuffle(data_set.get_dataset_size())
    data_set.channel_name = 'fasttext'

    return data_set
