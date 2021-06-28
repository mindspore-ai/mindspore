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

"""
CNN & DailyMail train dataset sampler
"""

import os
import sys
import shutil
import argparse
from random import random

from src.utils.tokenization import Tokenizer


def replace_split_word(read_path, output_path, tldr_str="TL;DR:", original_split='\t'):
    """
    append tldr str
    """
    with open(read_path, "r") as r, open(output_path, "a") as w:
        line = r.readline()
        while line:
            article = line[:line.find(original_split)] + ' ' + tldr_str + ' '
            ref = line[line.rfind(original_split) + 1:]
            w.write(article + ref)
            line = r.readline()


def sample(read_path, out_path, threshold=1.0, max_items=0xFFFFFFF):
    """
    sample function
    """
    cnt = 0
    total_cnt = 0
    with open(read_path, "r") as r, open(out_path, "a") as w:
        line = r.readline()
        while line:
            total_cnt += 1
            if cnt >= max_items:
                break
            if random() > threshold:
                line = r.readline()
                continue
            w.write(line)
            if (cnt + 1) % 3000 == 0:
                print("Now Processed Samples: {}, total: {}".format(cnt, total_cnt))
            cnt += 1
            line = r.readline()


def clip_article(input_path, out_path, hint, max_length, tokenizer_file_path):
    """
    clip article that the sample (article + summary) exceed max_length
    """
    tokenizer = Tokenizer(vocab_file=tokenizer_file_path + 'gpt2-vocab.json',
                          merge_file=tokenizer_file_path + 'gpt2-merges.txt')
    cnt = 0
    with open(input_path, "r") as r, open(out_path, "a+") as a:
        line = r.readline()
        while line:
            pos = line.rfind(hint)
            article = line[:pos]
            summary = line[pos:]
            if len(tokenizer.encode(line)) > max_length:
                l_article = tokenizer.encode(article)[:max_length - len(tokenizer.encode(summary))]
                article = tokenizer.decode(l_article) + " "
            if cnt % 1000 == 0:
                print(article + summary)
                print("==============================")
            cnt += 1
            a.write(article + summary)
            line = r.readline()


def sampler_dataset():
    """
    run CNN & DailyMail train dataset sampler
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="",
                        help="input file path")
    parser.add_argument("--output_path", type=str, default="",
                        help="out file path")
    parser.add_argument("--replace_hint", type=str, default="true")
    parser.add_argument("--sample", type=str, default="true",
                        help="do sample? true or false")
    parser.add_argument("--max_length", type=int, default=1022,
                        help="max seq_length of input_raw_dataset")
    parser.add_argument("--prob", type=float, default=0.25,
                        help="sample rate")
    parser.add_argument("--max_items", type=int, default=10000,
                        help="max number of document")
    parser.add_argument("--hint", type=str, default="TL;DR:",
                        help="hint text")
    parser.add_argument("--tokenizer_file_path", type=str, default="",
                        help="tokenizer helper file path")
    args = parser.parse_args()

    # temp_files, one for storing inputs in every stage, the other for storing middle results.
    temp_file_input = sys.path[0] + '/temp_file1_by_sampler_py.txt'
    temp_file_proc = sys.path[0] + '/temp_file2_by_sampler_py.txt'

    read_path = args.input_path
    output_path = args.output_path
    prob = args.prob
    max_items = args.max_items
    hint = args.hint
    max_length = args.max_length
    tokenizer_file_path = args.tokenizer_file_path
    split_str = '\t'

    shutil.copyfile(read_path, temp_file_input)
    clip_article(input_path=temp_file_input,
                 out_path=temp_file_proc,
                 hint=split_str,
                 max_length=max_length,
                 tokenizer_file_path=tokenizer_file_path)
    shutil.copyfile(temp_file_proc, temp_file_input)
    os.remove(temp_file_proc)

    if args.replace_hint.lower() == "true":
        replace_split_word(temp_file_input, temp_file_proc, hint, split_str)
        shutil.copyfile(temp_file_proc, temp_file_input)
        os.remove(temp_file_proc)

    if args.sample.lower() == "true":
        sample(temp_file_input, temp_file_proc, prob, max_items)
        shutil.copyfile(temp_file_proc, temp_file_input)
        os.remove(temp_file_proc)

    shutil.copyfile(temp_file_input, output_path)
    os.remove(temp_file_input)


if __name__ == "__main__":
    sampler_dataset()
