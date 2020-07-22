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
"""Generate News Crawl corpus dataset."""
import argparse

from src.utils import Dictionary
from src.utils.preprocess import create_pre_training_dataset

parser = argparse.ArgumentParser(description='Create News Crawl Pre-Training Dataset.')
parser.add_argument("--src_folder", type=str, default="", required=True,
                    help="Raw corpus folder.")
parser.add_argument("--existed_vocab", type=str, default="", required=True,
                    help="Existed vocab path.")
parser.add_argument("--mask_ratio", type=float, default=0.4, required=True,
                    help="Mask ratio.")
parser.add_argument("--output_folder", type=str, default="", required=True,
                    help="Dataset output path.")
parser.add_argument("--max_len", type=int, default=32, required=False,
                    help="Max length of sentences.")
parser.add_argument("--suffix", type=str, default="", required=False,
                    help="Add suffix to output file.")
parser.add_argument("--processes", type=int, default=2, required=False,
                    help="Size of processes pool.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    if not (args.src_folder and args.output_folder):
        raise ValueError("Please enter required params.")

    if not args.existed_vocab:
        raise ValueError("`--existed_vocab` is required.")

    vocab = Dictionary.load_from_persisted_dict(args.existed_vocab)

    create_pre_training_dataset(
        folder_path=args.src_folder,
        output_folder_path=args.output_folder,
        vocabulary=vocab,
        prefix="news.20", suffix=args.suffix,
        mask_ratio=args.mask_ratio,
        min_sen_len=10,
        max_sen_len=args.max_len,
        dataset_type="tfrecord",
        cores=args.processes
    )
    print(f" | Vocabulary size: {vocab.size}.")
