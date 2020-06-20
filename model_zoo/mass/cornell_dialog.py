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
"""Generate Cornell Movie Dialog dataset."""
import os
import argparse
from src.dataset import BiLingualDataLoader
from src.language_model import NoiseChannelLanguageModel
from src.utils import Dictionary

parser = argparse.ArgumentParser(description='Generate Cornell Movie Dialog dataset file.')
parser.add_argument("--src_folder", type=str, default="", required=True,
                    help="Raw corpus folder.")
parser.add_argument("--existed_vocab", type=str, default="", required=True,
                    help="Existed vocabulary.")
parser.add_argument("--train_prefix", type=str, default="train", required=False,
                    help="Prefix of train file.")
parser.add_argument("--test_prefix", type=str, default="test", required=False,
                    help="Prefix of test file.")
parser.add_argument("--valid_prefix", type=str, default=None, required=False,
                    help="Prefix of valid file.")
parser.add_argument("--noise_prob", type=float, default=0., required=False,
                    help="Add noise prob.")
parser.add_argument("--max_len", type=int, default=32, required=False,
                    help="Max length of sentence.")
parser.add_argument("--output_folder", type=str, default="", required=True,
                    help="Dataset output path.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    dicts = []
    train_src_file = ""
    train_tgt_file = ""
    test_src_file = ""
    test_tgt_file = ""
    valid_src_file = ""
    valid_tgt_file = ""
    for file in os.listdir(args.src_folder):
        if file.startswith(args.train_prefix) and "src" in file and file.endswith(".txt"):
            train_src_file = os.path.join(args.src_folder, file)
        elif file.startswith(args.train_prefix) and "tgt" in file and file.endswith(".txt"):
            train_tgt_file = os.path.join(args.src_folder, file)
        elif file.startswith(args.test_prefix) and "src" in file and file.endswith(".txt"):
            test_src_file = os.path.join(args.src_folder, file)
        elif file.startswith(args.test_prefix) and "tgt" in file and file.endswith(".txt"):
            test_tgt_file = os.path.join(args.src_folder, file)
        elif args.valid_prefix and file.startswith(args.valid_prefix) and "src" in file and file.endswith(".txt"):
            valid_src_file = os.path.join(args.src_folder, file)
        elif args.valid_prefix and file.startswith(args.valid_prefix) and "tgt" in file and file.endswith(".txt"):
            valid_tgt_file = os.path.join(args.src_folder, file)
        else:
            continue

    vocab = Dictionary.load_from_persisted_dict(args.existed_vocab)

    if train_src_file and train_tgt_file:
        BiLingualDataLoader(
            src_filepath=train_src_file,
            tgt_filepath=train_tgt_file,
            src_dict=vocab, tgt_dict=vocab,
            src_lang="en", tgt_lang="en",
            language_model=NoiseChannelLanguageModel(add_noise_prob=args.noise_prob),
            max_sen_len=args.max_len
        ).write_to_tfrecord(
            path=os.path.join(
                args.output_folder, "train_cornell_dialog.tfrecord"
            )
        )

    if test_src_file and test_tgt_file:
        BiLingualDataLoader(
            src_filepath=test_src_file,
            tgt_filepath=test_tgt_file,
            src_dict=vocab, tgt_dict=vocab,
            src_lang="en", tgt_lang="en",
            language_model=NoiseChannelLanguageModel(add_noise_prob=0.),
            max_sen_len=args.max_len
        ).write_to_tfrecord(
            path=os.path.join(
                args.output_folder, "test_cornell_dialog.tfrecord"
            )
        )

    if args.valid_prefix:
        BiLingualDataLoader(
            src_filepath=os.path.join(args.src_folder, valid_src_file),
            tgt_filepath=os.path.join(args.src_folder, valid_tgt_file),
            src_dict=vocab, tgt_dict=vocab,
            src_lang="en", tgt_lang="en",
            language_model=NoiseChannelLanguageModel(add_noise_prob=0.),
            max_sen_len=args.max_len
        ).write_to_tfrecord(
            path=os.path.join(
                args.output_folder, "valid_cornell_dialog.tfrecord"
            )
        )

    print(f" | Vocabulary size: {vocab.size}.")
