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

"""create mindrecord data for Children's Book Test task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import numpy as np

from mindspore.mindrecord import FileWriter
from src.utils.tokenization import Tokenizer


def create_instance(tokenizer, text, max_length=None, num_choice=None):
    """A single sample instance for cbt task."""
    text = text.replace(" \t ", "\t ")
    sentence = text.strip().split("\t")
    context_length = len(tokenizer.encode(sentence[0]))

    whole_sentence = sentence[0] + sentence[1]
    whole_sentence = whole_sentence.strip()
    assert whole_sentence != ""
    print(" | whole sentence: ", whole_sentence)
    ids = tokenizer.encode(whole_sentence)
    input_length = len(ids)
    pair_ids = None

    output = tokenizer.prepare_for_model(ids=ids,
                                         pair_ids=pair_ids,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         padding=True,
                                         truncate_direction="RIGHT",
                                         return_overflowing_tokens=False,
                                         return_attention_mask=True)

    output["length"] = [context_length + 1] + [input_length + 1]

    gold_answer_id = int(sentence[2])
    assert gold_answer_id < 10
    output["mc_labels"] = gold_answer_id

    for name, value in output.items():
        print(name)
        print(value)
        print("==================================")

    return output


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    input_mask = instance["attention_mask"]
    assert len(input_ids) == len(input_mask)
    length = instance["length"]  # list
    mc_labels = instance["mc_labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids)
    features["input_mask"] = np.asarray(input_mask)
    features["input_length"] = np.asarray(length)
    features["mc_labels"] = mc_labels

    writer.write_raw_data([features])
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, default="", help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, required=True, default="", help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help='The MindRecord file will be split into the number of partition. ')
    parser.add_argument("--max_length", type=int, required=True, help='Maximum sequence length. ')
    parser.add_argument("--num_choice", type=int, required=True, help='Number of choices. ')
    parser.add_argument("--vocab_file", type=str, required=True, default='', help='url of gpt2-vocab.json ')
    parser.add_argument("--merge_file", type=str, required=True, default='', help='url of gpt2-merges.txt ')
    args = parser.parse_args()

    tokenizer = Tokenizer(vocab_file=args.vocab_file, merge_file=args.merge_file)
    num_choice = args.num_choice

    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"input_ids": {"type": "int64", "shape": [-1]},
                   "input_mask": {"type": "int64", "shape": [-1]},
                   "input_length": {"type": "int64", "shape": [-1]},
                   "mc_labels": {"type": "int64"}
                   }
    writer.add_schema(data_schema, "cbt-schema")

    total_written = 0
    total_read = 0

    logging.info("***** Reading from  %s *****", input_file)
    with open(input_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            total_read += 1
            if total_read % 500 == 0:
                logging.info("%d ...", total_read)

            output = create_instance(tokenizer, line, args.max_length, num_choice)
            features = write_instance_to_file(writer, instance=output)
            total_written += 1

            if total_written <= 20:
                logging.info("***** Example *****")
                logging.info("input tokens: %s", tokenizer.decode(output["input_ids"][:-1]))
                logging.info("label tokens: %s", tokenizer.decode(output["input_ids"][1:]))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    logging.info("%s: %s", feature_name, feature)

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    main()
