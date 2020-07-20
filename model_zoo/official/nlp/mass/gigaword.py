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
"""Generate Gigaword dataset."""
import os
import argparse

from src.dataset import BiLingualDataLoader
from src.language_model import NoiseChannelLanguageModel
from src.utils import Dictionary

parser = argparse.ArgumentParser(description='Create Gigaword fine-tune Dataset.')
parser.add_argument("--train_src", type=str, default="", required=False,
                    help="train dataset source file path.")
parser.add_argument("--train_ref", type=str, default="", required=False,
                    help="train dataset reference file path.")
parser.add_argument("--test_src", type=str, default="", required=False,
                    help="test dataset source file path.")
parser.add_argument("--test_ref", type=str, default="", required=False,
                    help="test dataset reference file path.")
parser.add_argument("--noise_prob", type=float, default=0., required=False,
                    help="add noise prob.")
parser.add_argument("--existed_vocab", type=str, default="", required=False,
                    help="existed vocab path.")
parser.add_argument("--max_len", type=int, default=64, required=False,
                    help="max length of sentences.")
parser.add_argument("--output_folder", type=str, default="", required=True,
                    help="dataset output path.")
parser.add_argument("--format", type=str, default="tfrecord", required=False,
                    help="dataset format.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    vocab = Dictionary.load_from_persisted_dict(args.existed_vocab)

    if args.train_src and args.train_ref:
        train = BiLingualDataLoader(
            src_filepath=args.train_src,
            tgt_filepath=args.train_ref,
            src_dict=vocab, tgt_dict=vocab,
            src_lang="en", tgt_lang="en",
            language_model=NoiseChannelLanguageModel(add_noise_prob=args.noise_prob),
            max_sen_len=args.max_len
        )
        if "tf" in args.format.lower():
            train.write_to_tfrecord(
                path=os.path.join(args.output_folder, "gigaword_train_dataset.tfrecord")
            )
        else:
            train.write_to_mindrecord(
                path=os.path.join(args.output_folder, "gigaword_train_dataset.mindrecord")
            )

    if args.test_src and args.test_ref:
        test = BiLingualDataLoader(
            src_filepath=args.test_src,
            tgt_filepath=args.test_ref,
            src_dict=vocab, tgt_dict=vocab,
            src_lang="en", tgt_lang="en",
            language_model=NoiseChannelLanguageModel(add_noise_prob=0),
            max_sen_len=args.max_len
        )
        if "tf" in args.format.lower():
            test.write_to_tfrecord(
                path=os.path.join(args.output_folder, "gigaword_test_dataset.tfrecord")
            )
        else:
            test.write_to_mindrecord(
                path=os.path.join(args.output_folder, "gigaword_test_dataset.mindrecord")
            )

    print(f" | Vocabulary size: {vocab.size}.")
