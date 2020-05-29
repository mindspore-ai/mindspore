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
"""Create training instances for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import numpy as np
import src.tokenization as tokenization
from mindspore.mindrecord import FileWriter

class SampleInstance():
    """A single sample instance (sentence pair)."""

    def __init__(self, source_sos_tokens, source_eos_tokens, target_sos_tokens, target_eos_tokens):
        self.source_sos_tokens = source_sos_tokens
        self.source_eos_tokens = source_eos_tokens
        self.target_sos_tokens = target_sos_tokens
        self.target_eos_tokens = target_eos_tokens

    def __str__(self):
        s = ""
        s += "source sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_sos_tokens]))
        s += "source eos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_eos_tokens]))
        s += "target sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.target_sos_tokens]))
        s += "target eos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.target_eos_tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_file(writer, instance, tokenizer, max_seq_length):
    """Create files from `SampleInstance`s."""

    def _convert_ids_and_mask(input_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

    source_sos_ids, source_sos_mask = _convert_ids_and_mask(instance.source_sos_tokens)
    source_eos_ids, source_eos_mask = _convert_ids_and_mask(instance.source_eos_tokens)
    target_sos_ids, target_sos_mask = _convert_ids_and_mask(instance.target_sos_tokens)
    target_eos_ids, target_eos_mask = _convert_ids_and_mask(instance.target_eos_tokens)

    features = collections.OrderedDict()
    features["source_sos_ids"] = np.asarray(source_sos_ids)
    features["source_sos_mask"] = np.asarray(source_sos_mask)
    features["source_eos_ids"] = np.asarray(source_eos_ids)
    features["source_eos_mask"] = np.asarray(source_eos_mask)
    features["target_sos_ids"] = np.asarray(target_sos_ids)
    features["target_sos_mask"] = np.asarray(target_sos_mask)
    features["target_eos_ids"] = np.asarray(target_eos_ids)
    features["target_eos_mask"] = np.asarray(target_eos_mask)

    writer.write_raw_data([features])
    return features

def create_training_instance(source_words, target_words, max_seq_length, clip_to_max_len):
    """Creates `SampleInstance`s for a single sentence pair."""
    EOS = "</s>"
    SOS = "<s>"

    if len(source_words) >= max_seq_length or len(target_words) >= max_seq_length:
        if clip_to_max_len:
            print("####lalalal")
            source_words = source_words[:min([len(source_words, max_seq_length-1)])]
            target_words = target_words[:min([len(target_words, max_seq_length-1)])]
        else:
            return None

    source_sos_tokens = [SOS] + source_words
    source_eos_tokens = source_words + [EOS]
    target_sos_tokens = [SOS] + target_words
    target_eos_tokens = target_words + [EOS]

    instance = SampleInstance(
        source_sos_tokens=source_sos_tokens,
        source_eos_tokens=source_eos_tokens,
        target_sos_tokens=target_sos_tokens,
        target_eos_tokens=target_eos_tokens)
    return instance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help='Input raw text file (or comma-separated list of files).')
    parser.add_argument("--output_file", type=str, required=True, help='Output MindRecord file.')
    parser.add_argument("--num_splits", type=int, default=16,
                        help='The MindRecord file will be split into the number of partition.')
    parser.add_argument("--vocab_file", type=str, required=True,
                        help='The vocabulary file that the Transformer model was trained on.')
    parser.add_argument("--clip_to_max_len", type=bool, default=False,
                        help='clip sequences to maximum sequence length.')
    parser.add_argument("--max_seq_length", type=int, default=128, help='Maximum sequence length.')
    args = parser.parse_args()

    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=args.vocab_file)

    input_files = []
    for input_pattern in args.input_file.split(","):
        input_files.append(input_pattern)

    logging.info("*** Reading from input files ***")
    for input_file in input_files:
        logging.info("  %s", input_file)

    output_file = args.output_file
    logging.info("*** Writing to output files ***")
    logging.info("  %s", output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"source_sos_ids": {"type": "int64", "shape": [-1]},
                   "source_sos_mask": {"type": "int64", "shape": [-1]},
                   "source_eos_ids": {"type": "int64", "shape": [-1]},
                   "source_eos_mask": {"type": "int64", "shape": [-1]},
                   "target_sos_ids": {"type": "int64", "shape": [-1]},
                   "target_sos_mask": {"type": "int64", "shape": [-1]},
                   "target_eos_ids": {"type": "int64", "shape": [-1]},
                   "target_eos_mask": {"type": "int64", "shape": [-1]}
                   }
    writer.add_schema(data_schema, "tranformer hisi")

    total_written = 0
    total_read = 0

    for input_file in input_files:
        logging.info("*** Reading from   %s ***", input_file)
        with open(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break

                total_read += 1
                if total_read % 100000 == 0:
                    logging.info("%d ...", total_read)

                source_line, target_line = line.strip().split("\t")
                source_tokens = tokenizer.tokenize(source_line)
                target_tokens = tokenizer.tokenize(target_line)

                if len(source_tokens) >= args.max_seq_length or len(target_tokens) >= args.max_seq_length:
                    logging.info("ignore long sentence!")
                    continue

                instance = create_training_instance(source_tokens, target_tokens, args.max_seq_length,
                                                    clip_to_max_len=args.clip_to_max_len)
                if instance is None:
                    continue

                features = write_instance_to_file(writer, instance, tokenizer, args.max_seq_length)
                total_written += 1

                if total_written <= 20:
                    logging.info("*** Example ***")
                    logging.info("source tokens: %s", " ".join(
                        [tokenization.printable_text(x) for x in instance.source_eos_tokens]))
                    logging.info("target tokens: %s", " ".join(
                        [tokenization.printable_text(x) for x in instance.target_sos_tokens]))

                    for feature_name in features.keys():
                        feature = features[feature_name]
                        logging.info("%s: %s", feature_name, feature)

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    main()
