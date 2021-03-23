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

"""
data convert to mindrecord file.
"""

import os
import argparse
import numpy as np
import dataset as data
from tokenizer import FullTokenizer
from data_util import Tuple, Pad, Stack
from mindspore.mindrecord import FileWriter
TASK_CLASSES = {
    'udc': data.UDCv1,
    'dstc2': data.DSTC2,
    'atis_slot': data.ATIS_DSF,
    'atis_intent': data.ATIS_DID,
    'mrda': data.MRDA,
    'swda': data.SwDA,
}

def data_save_to_file(data_file_path=None, vocab_file_path='bert-base-uncased-vocab.txt', \
        output_path=None, task_name=None, mode="train", max_seq_length=128):
    """data save to mindrecord file."""
    MINDRECORD_FILE_PATH = output_path + task_name+"/" + task_name + "_" + mode + ".mindrecord"
    if not os.path.exists(output_path + task_name):
        os.makedirs(output_path + task_name)
    if os.path.exists(MINDRECORD_FILE_PATH):
        os.remove(MINDRECORD_FILE_PATH)
        os.remove(MINDRECORD_FILE_PATH + ".db")
    dataset_class = TASK_CLASSES[task_name]
    tokenizer = FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)
    dataset = dataset_class(data_file_path+task_name, mode=mode)
    applid_data = []
    datalist = []
    print(task_name + " " + mode + " data process begin")
    dataset_len = len(dataset)
    if args.task_name == 'atis_slot':
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0),  # input
            Pad(axis=0, pad_val=0),  # mask
            Pad(axis=0, pad_val=0),  # segment
            Pad(axis=0, pad_val=0, dtype='int64')  # label
        ): fn(samples)
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0),  # input
            Pad(axis=0, pad_val=0),  # mask
            Pad(axis=0, pad_val=0),  # segment
            Stack(dtype='int64')  # label
        ): fn(samples)
    for idx, example in enumerate(dataset):
        if idx % 1000 == 0:
            print("Reading example %d of %d" % (idx, dataset_len))
        data_example = dataset_class.convert_example(example=example, \
                tokenizer=tokenizer, max_seq_length=max_seq_length)
        applid_data.append(data_example)

    applid_data = batchify_fn(applid_data)
    input_ids, input_mask, segment_ids, label_ids = applid_data

    for idx in range(dataset_len):
        if idx % 1000 == 0:
            print("Processing example %d of %d" % (idx, dataset_len))
        sample = {
            "input_ids": np.array(input_ids[idx], dtype=np.int64),
            "input_mask": np.array(input_mask[idx], dtype=np.int64),
            "segment_ids": np.array(segment_ids[idx], dtype=np.int64),
            "label_ids": np.array([label_ids[idx]], dtype=np.int64),
        }
        datalist.append(sample)

    print(task_name + " " + mode + " data process end")
    writer = FileWriter(file_name=MINDRECORD_FILE_PATH, shard_num=1)
    nlp_schema = {
        "input_ids": {"type": "int64", "shape": [-1]},
        "input_mask": {"type": "int64", "shape": [-1]},
        "segment_ids": {"type": "int64", "shape": [-1]},
        "label_ids": {"type": "int64", "shape": [-1]},
    }
    writer.add_schema(nlp_schema, "proprocessed classification dataset")
    writer.write_raw_data(datalist)
    writer.commit()
    print("write success")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run classifier")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The directory where the dataset will be load.")
    parser.add_argument(
        "--vocab_file_dir",
        default=None,
        type=str,
        help="The directory where the vocab will be load.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The directory where the mindrecord dataset file will be save.")
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization for trainng. ")
    parser.add_argument(
        "--eval_max_seq_len",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization for trainng. ")

    args = parser.parse_args()
    if args.eval_max_seq_len is None:
        args.eval_max_seq_len = args.max_seq_len
    data_save_to_file(data_file_path=args.data_dir, vocab_file_path=args.vocab_file_dir, output_path=args.output_dir, \
            task_name=args.task_name, mode="train", max_seq_length=args.max_seq_len)
    data_save_to_file(data_file_path=args.data_dir, vocab_file_path=args.vocab_file_dir, output_path=args.output_dir, \
            task_name=args.task_name, mode="dev", max_seq_length=args.eval_max_seq_len)
    data_save_to_file(data_file_path=args.data_dir, vocab_file_path=args.vocab_file_dir, output_path=args.output_dir, \
            task_name=args.task_name, mode="test", max_seq_length=args.eval_max_seq_len)
