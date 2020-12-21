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
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as deC

def load_dataset(dataset_path,
                 batch_size,
                 epoch_count=1,
                 rank_size=1,
                 rank_id=0,
                 bucket=None,
                 shuffle=True):
    """dataset loader"""
    def batch_per_bucket(bucket_length, input_file):
        input_file = input_file +'/train_dataset_bs_' + str(bucket_length) + '.mindrecord'
        if not input_file:
            raise FileNotFoundError("input file parameter must not be empty.")

        ds = de.MindDataset(input_file,
                            columns_list=['src_tokens', 'src_tokens_length', 'label_idx'],
                            shuffle=shuffle,
                            num_shards=rank_size,
                            shard_id=rank_id,
                            num_parallel_workers=8)
        ori_dataset_size = ds.get_dataset_size()
        print(f"Dataset size: {ori_dataset_size}")
        repeat_count = epoch_count
        type_cast_op = deC.TypeCast(mstype.int32)
        ds = ds.map(operations=type_cast_op, input_columns="src_tokens")
        ds = ds.map(operations=type_cast_op, input_columns="src_tokens_length")
        ds = ds.map(operations=type_cast_op, input_columns="label_idx")

        ds = ds.rename(input_columns=['src_tokens', 'src_tokens_length', 'label_idx'],
                       output_columns=['src_token_text', 'src_tokens_text_length', 'label_idx_tag'])
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.repeat(repeat_count)
        return ds
    for i, _ in enumerate(bucket):
        bucket_len = bucket[i]
        ds_per = batch_per_bucket(bucket_len, dataset_path)
        if i == 0:
            ds = ds_per
        else:
            ds = ds + ds_per
    ds = ds.shuffle(ds.get_dataset_size())
    ds.channel_name = 'fasttext'

    return ds
