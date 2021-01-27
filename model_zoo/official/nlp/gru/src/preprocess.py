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
'''Dataset preprocess'''
import os
import argparse
from collections import Counter
from nltk.tokenize import word_tokenize

def create_tokenized_sentences(input_files, language):
    '''
    Create tokenized sentences files.

    Args:
        input_files: input files.
        language: text language
    '''
    sentence = []
    total_lines = open(input_files, "r").read().splitlines()
    for line in total_lines:
        line = line.strip('\r\n ')
        line = line.lower()
        tokenize_sentence = word_tokenize(line, language)
        str_sentence = " ".join(tokenize_sentence)
        sentence.append(str_sentence)
    tokenize_file = input_files + ".tok"
    f = open(tokenize_file, "w")
    for line in sentence:
        f.write(line)
        f.write("\n")
    f.close()

def get_dataset_vocab(text_file, vocab_file):
    '''
    Create dataset vocab files.

    Args:
        text_file: dataset text files.
        vocab_file: vocab file
    '''
    counter = Counter()
    text_lines = open(text_file, "r").read().splitlines()
    for line in text_lines:
        for word in line.strip('\r\n ').split(' '):
            if word:
                counter[word] += 1
    vocab = open(vocab_file, "w")
    basic_label = ["<unk>", "<pad>", "<sos>", "<eos>"]
    for label in basic_label:
        vocab.write(label + "\n")
    for key, f in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        if f < 2:
            continue
        vocab.write(key + "\n")
    vocab.close()

def MergeText(root_dir, file_list, output_file):
    '''
    Merge text files together.

    Args:
        root_dir: root dir
        file_list: dataset files list.
        output_file: output file after merge
    '''
    output_file = os.path.join(root_dir, output_file)
    f_output = open(output_file, "w")
    for file_name in file_list:
        text_path = os.path.join(root_dir, file_name) + ".tok"
        f = open(text_path)
        f_output.write(f.read() + "\n")
    f_output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gru_dataset')
    parser.add_argument("--dataset_path", type=str, default="", help="Dataset path, default: f`sns.")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    src_file_list = ["train.de", "test.de", "val.de"]
    dst_file_list = ["train.en", "test.en", "val.en"]
    for file in src_file_list:
        file_path = os.path.join(dataset_path, file)
        create_tokenized_sentences(file_path, "english")
    for file in dst_file_list:
        file_path = os.path.join(dataset_path, file)
        create_tokenized_sentences(file_path, "german")
    src_all_file = "all.de.tok"
    dst_all_file = "all.en.tok"
    MergeText(dataset_path, src_file_list, src_all_file)
    MergeText(dataset_path, dst_file_list, dst_all_file)
    src_vocab = os.path.join(dataset_path, "vocab.de")
    dst_vocab = os.path.join(dataset_path, "vocab.en")
    get_dataset_vocab(os.path.join(dataset_path, src_all_file), src_vocab)
    get_dataset_vocab(os.path.join(dataset_path, dst_all_file), dst_vocab)
