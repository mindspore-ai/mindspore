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
"""Tokenizer."""
import os
import argparse
from typing import Callable
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Corpus tokenizer which text file must end with `.txt`.')
parser.add_argument("--corpus_folder", type=str, default="", required=True,
                    help="Corpus folder path, if multi-folders are provided, use ',' split folders.")
parser.add_argument("--output_folder", type=str, default="", required=True,
                    help="Output folder path.")
parser.add_argument("--tokenizer", type=str, default="nltk", required=False,
                    help="Tokenizer to be used, nltk or jieba, if nltk is not installed fully, "
                         "use jieba instead.")
parser.add_argument("--pool_size", type=int, default=2, required=False,
                    help="Processes pool size.")

TOKENIZER = Callable


def create_tokenized_sentences(file_path, tokenized_file):
    """
    Create tokenized sentences.

    Args:
        file_path (str): Text file.
        tokenized_file (str): Output file.
    """
    global TOKENIZER

    print(f" | Processing {file_path}.")
    tokenized_sen = []
    with open(file_path, "r") as file:
        for sen in file:
            tokens = TOKENIZER(sen)
            tokens = [t for t in tokens if t != " "]
            if len(tokens) > 175:
                continue
            tokenized_sen.append(" ".join(tokens) + "\n")

    with open(tokenized_file, "w") as file:
        file.writelines(tokenized_sen)
    print(f" | Wrote to {tokenized_file}.")


def tokenize():
    """Tokenizer."""
    global TOKENIZER

    args, _ = parser.parse_known_args()
    src_folder = args.corpus_folder.split(",")

    try:
        from nltk.tokenize import word_tokenize

        TOKENIZER = word_tokenize
    except (ImportError, ModuleNotFoundError, LookupError):
        try:
            import jieba
        except Exception as e:
            raise e

        print(" | NLTK is not found, use jieba instead.")
        TOKENIZER = jieba.cut

    if args.tokenizer == "jieba":
        import jieba
        TOKENIZER = jieba.cut

    pool = Pool(args.pool_size)
    for folder in src_folder:
        for file in os.listdir(folder):
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(folder, file)
            out_path = os.path.join(args.output_folder, file.replace(".txt", "_tokenized.txt"))
            pool.apply_async(create_tokenized_sentences, (file_path, out_path,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    tokenize()
