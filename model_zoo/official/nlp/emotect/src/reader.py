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
'''
Dataset reader for preprocessing and converting dataset into MindRecord.
'''

import io
import argparse
import collections
import six
import numpy as np
from mindspore.mindrecord import FileWriter
from mindspore.log import logging
from tokenizer import FullTokenizer


def csv_reader(fd, delimiter='\t'):
    """
    csv 文件读取
    """
    def gen():
        for i in fd:
            slots = i.rstrip('\n').split(delimiter)
            if len(slots) == 1:
                yield (slots,)
            else:
                yield slots
    return gen()

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            text = text
        elif isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            text = text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            text = text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
    return text

class BaseReader:
    """BaseReader for classify and sequence labeling task"""

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens

        np.random.seed(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
            headers = next(reader)
            Example = collections.namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if "text_b" in example._fields:
            text_b = convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = collections.namedtuple(
            'Record',
            ['input_ids', 'input_mask', 'segment_ids', 'label_id'])

        record = Record(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
        return record

    def get_num_examples(self, input_file):
        """return total number of examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def get_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return examples

    def file_based_convert_examples_to_features(self, input_file, output_file):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._read_tsv(input_file)

        writer = FileWriter(file_name=output_file, shard_num=1)
        nlp_schema = {
            "input_ids": {"type": "int64", "shape": [-1]},
            "input_mask": {"type": "int64", "shape": [-1]},
            "segment_ids": {"type": "int64", "shape": [-1]},
            "label_ids": {"type": "int64", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        data = []
        for index, example in enumerate(examples):
            if index % 10000 == 0:
                logging.info("Writing example %d of %d" % (index, len(examples)))
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer)
            sample = {
                "input_ids": np.array(record.input_ids, dtype=np.int64),
                "input_mask": np.array(record.input_mask, dtype=np.int64),
                "segment_ids": np.array(record.segment_ids, dtype=np.int64),
                "label_ids": np.array([record.label_id], dtype=np.int64),
            }
            data.append(sample)
        writer.write_raw_data(data)
        writer.commit()

class ClassifyReader(BaseReader):
    """ClassifyReader"""

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = collections.namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text.replace(' ', '')
                example = Example(*line)
                examples.append(example)
            return examples

def main():
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
    parser.add_argument("--label_map_config", type=str, default=None, help="label mapping config file")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed number")
    parser.add_argument("--input_file", type=str, default="", help="raw data file")
    parser.add_argument("--output_file", type=str, default="", help="minddata file")
    args_opt = parser.parse_args()
    reader = ClassifyReader(
        vocab_path=args_opt.vocab_path,
        label_map_config=args_opt.label_map_config,
        max_seq_len=args_opt.max_seq_len,
        do_lower_case=args_opt.do_lower_case,
        random_seed=args_opt.random_seed
    )
    reader.file_based_convert_examples_to_features(input_file=args_opt.input_file, output_file=args_opt.output_file)

if __name__ == "__main__":
    main()
