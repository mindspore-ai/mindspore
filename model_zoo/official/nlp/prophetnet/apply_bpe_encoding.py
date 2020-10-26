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
"""Apply bpe script."""
import os
import argparse
from multiprocessing import Pool, cpu_count

from src.utils import Dictionary
from src.utils import bpe_encode

parser = argparse.ArgumentParser(description='Apply BPE.')
parser.add_argument("--codes", type=str, default="", required=True,
                    help="bpe codes path.")
parser.add_argument("--src_folder", type=str, default="", required=True,
                    help="raw corpus folder.")
parser.add_argument("--output_folder", type=str, default="", required=True,
                    help="encoded corpus output path.")
parser.add_argument("--prefix", type=str, default="", required=False,
                    help="Prefix of text file.")
parser.add_argument("--vocab_path", type=str, default="", required=True,
                    help="Generated vocabulary output path.")
parser.add_argument("--threshold", type=int, default=None, required=False,
                    help="Filter out words that frequency is lower than threshold.")
parser.add_argument("--processes", type=int, default=2, required=False,
                    help="Number of processes to use.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    if not (args.codes and args.src_folder and args.output_folder):
        raise ValueError("Please enter required params.")

    source_folder = args.src_folder
    output_folder = args.output_folder
    codes = args.codes

    if not os.path.exists(codes):
        raise FileNotFoundError("`--codes` is not existed.")
    if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
        raise ValueError("`--src_folder` must be a dir and existed.")
    if not os.path.exists(output_folder) or not os.path.isdir(output_folder):
        raise ValueError("`--output_folder` must be a dir and existed.")
    if not isinstance(args.prefix, str) or len(args.prefix) > 128:
        raise ValueError("`--prefix` must be a str and len <= 128.")
    if not isinstance(args.processes, int):
        raise TypeError("`--processes` must be an integer.")

    available_dict = []
    args_groups = []
    for file in os.listdir(source_folder):
        if args.prefix and not file.startswith(args.prefix):
            continue
        if file.endswith(".txt"):
            output_path = os.path.join(output_folder, file.replace(".txt", "_bpe.txt"))
            dict_path = os.path.join(output_folder, file.replace(".txt", ".dict"))
            available_dict.append(dict_path)
            args_groups.append((codes, os.path.join(source_folder, file),
                                output_path, dict_path))

    kernel_size = 1 if args.processes <= 0 else args.processes
    kernel_size = min(kernel_size, cpu_count())
    pool = Pool(kernel_size)
    for arg in args_groups:
        pool.apply_async(bpe_encode, args=arg)
    pool.close()
    pool.join()

    vocab = Dictionary.load_from_text(available_dict)
    if args.threshold is not None:
        vocab = vocab.shrink(args.threshold)
    vocab.persistence(args.vocab_path)
    print(f" | Vocabulary Size: {len(vocab)}")
