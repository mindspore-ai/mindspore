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
"""make finetune-mindrecord dataset."""
import os
import json
import argparse
import numpy as np
from tqdm import trange

from mindspore.mindrecord import FileWriter

from tokenizer_cpm import CPMTokenizer


class CHIDDataset():
    """Dataset define for ChId."""
    def __init__(self, data_path, tokenizer):
        self.pad_id = tokenizer.pad_id
        with open(data_path, "r") as f:
            self.cand_ids, data_cpm = json.load(f)

        self.samples, self.sizes = self.process(data_cpm)
        self.max_size = max(self.sizes)

    def process(self, process_data, num_samples=10):
        """Dataset process."""
        sizes = []
        samples = []
        for d in process_data:
            loss_mask = [0] * (len(d["sent"]) - 2) + [1]

            samples.append((
                d["sent"][:-1],  # ids for the tokenized sentence
                loss_mask,  # mask of the loss
                d["sent"][1:],  # token labels of each sentence
                d["truth"],  # labels if each sentence, should be an integer in [0, 9]
            ))
            sizes.append(len(d["sent"]) - 1)
        return samples, sizes

    def _pad_process(self, input_ids, loss_mask, labels):
        """Dataset padding process."""
        pad_input_ids = np.ones(shape=(self.max_size), dtype=np.int64) * self.pad_id
        pad_loss_mask = np.zeros(shape=(self.max_size)) * 1.0
        pad_labels = np.ones(shape=(self.max_size), dtype=np.int64) * self.pad_id

        pad_input_ids[:len(input_ids)] = input_ids
        pad_loss_mask[:len(loss_mask)] = loss_mask
        pad_labels[:len(labels)] = labels

        return pad_input_ids, pad_loss_mask, pad_labels

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        input_ids, loss_mask, labels, truth = self.samples[idx]
        pad_input_ids, pad_loss_mask, pad_labels = self._pad_process(input_ids, loss_mask, labels)
        sample = {
            "truth": truth,
            "input_ids": pad_input_ids,
            "loss_mask": pad_loss_mask,
            "labels": pad_labels,
            "size": self.sizes[idx]
        }
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPM dataset.')
    parser.add_argument("--vocab_path", type=str, required=False,
                        default="./vocab",
                        help="the tokenizer vocab path.")

    parser.add_argument("--data_file", type=str, required=False,
                        default="./preprocessed/train.json",
                        help="finetune train dataset files.")

    parser.add_argument("--output_path", type=str, required=False,
                        default="./output/train.mindrecord",
                        help="mindrecord dataset output path.")
    args = parser.parse_args()

    # get the tokenizer
    tokenizer_cpm = CPMTokenizer(os.path.join(args.vocab_path, 'vocab.json'),
                                 os.path.join(args.vocab_path, 'chinese_vocab.model'))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    chidDataset = CHIDDataset(args.data_file, tokenizer_cpm)
    chid_schema = {"truth": {"type": "int64"},
                   "input_ids": {"type": "int64", "shape": [-1]},
                   "loss_mask": {"type": "float64", "shape": [-1]},
                   "labels": {"type": "int64", "shape": [-1]},
                   "size": {"type": "int64"}}

    writer = FileWriter(file_name=args.output_path)
    writer.add_schema(chid_schema, "preprocessed chid dataset")
    data = []
    for i in trange(len(chidDataset)):
        data.append(chidDataset[i])
        if i % 100 == 0:
            writer.write_raw_data(data)
            data = []

if data:
    writer.write_raw_data(data)

writer.commit()
print("transform mindrecord successfully, refer: {}".format(args.output_path))
