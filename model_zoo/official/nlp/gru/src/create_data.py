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
"""Create training instances for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import collections
import logging
import numpy as np
import tokenization
from mindspore.mindrecord import FileWriter

class SampleInstance():
    """A single sample instance (sentence pair)."""

    def __init__(self, source_tokens, target_tokens):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens

    def __str__(self):
        s = ""
        s += "source_tokens: %s\n" % (" ".join(
            [tokenization.convert_to_printable(x) for x in self.source_tokens]))
        s += "target tokens: %s\n" % (" ".join(
            [tokenization.convert_to_printable(x) for x in self.target_tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def get_instance_features(instance, tokenizer_src, tokenizer_trg, max_seq_length, bucket):
    """Get features from `SampleInstance`s."""
    def _find_bucket_length(source_tokens, target_tokens):
        source_ids = tokenizer_src.convert_tokens_to_ids(source_tokens)
        target_ids = tokenizer_trg.convert_tokens_to_ids(target_tokens)
        num = max(len(source_ids), len(target_ids))
        assert num <= bucket[-1]
        for index in range(1, len(bucket)):
            if bucket[index - 1] < num <= bucket[index]:
                return bucket[index]
        return bucket[0]

    def _convert_ids_and_mask(tokenizer, input_tokens, seq_max_bucket_length):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < seq_max_bucket_length:
            input_ids.append(1)
            input_mask.append(0)

        assert len(input_ids) == seq_max_bucket_length
        assert len(input_mask) == seq_max_bucket_length

        return input_ids, input_mask

    seq_max_bucket_length = _find_bucket_length(instance.source_tokens, instance.target_tokens)
    source_ids, source_mask = _convert_ids_and_mask(tokenizer_src, instance.source_tokens, seq_max_bucket_length)
    target_ids, target_mask = _convert_ids_and_mask(tokenizer_trg, instance.target_tokens, seq_max_bucket_length)

    features = collections.OrderedDict()
    features["source_ids"] = np.asarray(source_ids)
    features["source_mask"] = np.asarray(source_mask)
    features["target_ids"] = np.asarray(target_ids)
    features["target_mask"] = np.asarray(target_mask)

    return features, seq_max_bucket_length

def create_training_instance(source_words, target_words, max_seq_length, clip_to_max_len):
    """Creates `SampleInstance`s for a single sentence pair."""
    EOS = "<eos>"
    SOS = "<sos>"

    if len(source_words) >= max_seq_length-1 or len(target_words) >= max_seq_length-1:
        if clip_to_max_len:
            source_words = source_words[:min([len(source_words, max_seq_length-2)])]
            target_words = target_words[:min([len(target_words, max_seq_length-2)])]
        else:
            return None
    source_tokens = [SOS] + source_words + [EOS]
    target_tokens = [SOS] + target_words + [EOS]
    instance = SampleInstance(
        source_tokens=source_tokens,
        target_tokens=target_tokens)
    return instance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help='Input raw text file (or comma-separated list of files).')
    parser.add_argument("--output_file", type=str, required=True, help='Output MindRecord file.')
    parser.add_argument("--num_splits", type=int, default=16,
                        help='The MindRecord file will be split into the number of partition.')
    parser.add_argument("--src_vocab_file", type=str, required=True,
                        help='The vocabulary file that the Transformer model was trained on.')
    parser.add_argument("--trg_vocab_file", type=str, required=True,
                        help='The vocabulary file that the Transformer model was trained on.')
    parser.add_argument("--clip_to_max_len", type=ast.literal_eval, default=False,
                        help='clip sequences to maximum sequence length.')
    parser.add_argument("--max_seq_length", type=int, default=32, help='Maximum sequence length.')
    parser.add_argument("--bucket", type=ast.literal_eval, default=[32],
                        help='bucket sequence length')
    args = parser.parse_args()
    tokenizer_src = tokenization.WhiteSpaceTokenizer(vocab_file=args.src_vocab_file)
    tokenizer_trg = tokenization.WhiteSpaceTokenizer(vocab_file=args.trg_vocab_file)
    input_files = []
    for input_pattern in args.input_file.split(","):
        input_files.append(input_pattern)
    logging.info("*** Read from input files ***")
    output_file = args.output_file
    logging.info("*** Write to output files ***")
    logging.info("  %s", output_file)
    total_written = 0
    total_read = 0
    feature_dict = {}
    for i in args.bucket:
        feature_dict[i] = []
    for input_file in input_files:
        logging.info("*** Reading from   %s ***", input_file)
        with open(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                total_read += 1
                if total_read % 100000 == 0:
                    logging.info("Read %d ...", total_read)
                if line.strip() == "":
                    continue
                source_line, target_line = line.strip().split("\t")
                source_tokens = tokenizer_src.tokenize(source_line)
                target_tokens = tokenizer_trg.tokenize(target_line)
                if len(source_tokens) >= args.max_seq_length or len(target_tokens) >= args.max_seq_length:
                    logging.info("ignore long sentence!")
                    continue
                instance = create_training_instance(source_tokens, target_tokens, args.max_seq_length,
                                                    clip_to_max_len=args.clip_to_max_len)
                if instance is None:
                    continue
                features, seq_max_bucket_length = get_instance_features(instance, tokenizer_src, tokenizer_trg,
                                                                        args.max_seq_length, args.bucket)
                for key in feature_dict:
                    if key == seq_max_bucket_length:
                        feature_dict[key].append(features)
                if total_read <= 10:
                    logging.info("*** Example ***")
                    logging.info("source tokens: %s", " ".join(
                        [tokenization.convert_to_printable(x) for x in instance.source_tokens]))
                    logging.info("target tokens: %s", " ".join(
                        [tokenization.convert_to_printable(x) for x in instance.target_tokens]))

                    for feature_name in features.keys():
                        feature = features[feature_name]
                        logging.info("%s: %s", feature_name, feature)
    for i in args.bucket:
        if args.num_splits == 1:
            output_file_name = output_file + '_' + str(i)
        else:
            output_file_name = output_file + '_' + str(i) + '_'
        writer = FileWriter(output_file_name, args.num_splits)
        data_schema = {"source_ids": {"type": "int64", "shape": [-1]},
                       "source_mask": {"type": "int64", "shape": [-1]},
                       "target_ids": {"type": "int64", "shape": [-1]},
                       "target_mask": {"type": "int64", "shape": [-1]}
                       }
        writer.add_schema(data_schema, "gru")
        features_ = feature_dict[i]
        logging.info("Bucket length %d has %d samples, start writing...", i, len(features_))
        for item in features_:
            writer.write_raw_data([item])
            total_written += 1
        writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
