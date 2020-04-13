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
"""create MindDataset by MindRecord"""
import argparse
import mindspore.dataset as ds

def create_dataset(data_file):
    """create MindDataset"""
    num_readers = 4
    data_set = ds.MindDataset(dataset_file=data_file, num_parallel_workers=num_readers, shuffle=True)
    index = 0
    for item in data_set.create_dict_iterator():
        # print("example {}: {}".format(index, item))
        print("example {}: input_ids: {}".format(index, item['input_ids']))
        print("example {}: input_mask: {}".format(index, item['input_mask']))
        print("example {}: segment_ids: {}".format(index, item['segment_ids']))
        print("example {}: masked_lm_positions: {}".format(index, item['masked_lm_positions']))
        print("example {}: masked_lm_ids: {}".format(index, item['masked_lm_ids']))
        print("example {}: masked_lm_weights: {}".format(index, item['masked_lm_weights']))
        print("example {}: next_sentence_labels: {}".format(index, item['next_sentence_labels']))
        index += 1
        if index % 1000 == 0:
            print("read rows: {}".format(index))
    print("total rows: {}".format(index))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help='Input mindreord file')
    args = parser.parse_args()

    create_dataset(args.input_file)
