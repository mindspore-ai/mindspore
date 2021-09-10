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
import json
from collections import namedtuple
import numpy as np
from mindspore.mindrecord import FileWriter
from mindspore.log import logging
from src.tokenizer import FullTokenizer, convert_to_unicode, tokenize_chinese_chars

def csv_reader(fd, delimiter='\t'):
    """
    read csv file
    """
    def gen():
        for i in fd:
            slots = i.rstrip('\n').split(delimiter)
            if len(slots) == 1:
                yield (slots,)
            else:
                yield slots
    return gen()

class BaseReader:
    """BaseReader for classify and sequence labeling task"""

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 random_seed=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab

        np.random.seed(random_seed)

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
            Example = namedtuple('Example', headers)

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
        token_type_id = []
        tokens.append("[CLS]")
        token_type_id.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_id.append(0)
        tokens.append("[SEP]")
        token_type_id.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_id.append(1)
            tokens.append("[SEP]")
            token_type_id.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            token_type_id.append(0)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['input_ids', 'input_mask', 'token_type_id', 'label_id'])

        record = Record(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_id=token_type_id,
            label_id=label_id)
        return record

    def get_num_examples(self, input_file):
        """return total number of examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def get_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return examples

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
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text.replace(' ', '')
                example = Example(*line)
                examples.append(example)
            return examples

    def file_based_convert_examples_to_features(self, input_file, output_file, shard_num, is_training):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._read_tsv(input_file)

        writer = FileWriter(file_name=output_file, shard_num=shard_num)
        nlp_schema = {
            "input_ids": {"type": "int64", "shape": [-1]},
            "input_mask": {"type": "int64", "shape": [-1]},
            "token_type_id": {"type": "int64", "shape": [-1]},
            "label_ids": {"type": "int64", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        data = []
        for index, example in enumerate(examples):
            if index % 1000 == 0:
                print("Writing example %d of %d" % (index, len(examples)))
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer)
            sample = {
                "input_ids": np.array(record.input_ids, dtype=np.int64),
                "input_mask": np.array(record.input_mask, dtype=np.int64),
                "token_type_id": np.array(record.token_type_id, dtype=np.int64),
                "label_ids": np.array([record.label_id], dtype=np.int64),
            }
            data.append(sample)
        writer.write_raw_data(data)
        writer.commit()

class SequenceLabelingReader(BaseReader):
    """sequence labeling reader"""
    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        tokens = convert_to_unicode(example.text_a).split(u"")
        labels = convert_to_unicode(example.label).split(u"")
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id] + [self.label_map[label] for label in labels] + [no_entity_id]
        input_mask = [1] * len(input_ids)
        token_type_id = [0] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            token_type_id.append(0)
            label_ids.append(no_entity_id)

        Record = namedtuple(
            'Record',
            ['input_ids', 'input_mask', 'token_type_id', 'label_ids'])

        record = Record(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_id=token_type_id,
            label_ids=label_ids)
        return record

    def _reseg_token_label(self, tokens, labels, tokenizer):
        """resegmentation toke label"""
        ret_tokens, ret_labels = [], []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if not sub_token:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))
            elif label.startswith("S-"):
                b_laebl = "B-" + label[2:]
                e_label = "E-" + label[2:]
                i_label = "I-" + label[2:]
                ret_labels.extend([b_laebl] + [i_label] * (len(sub_token) - 2) + [e_label])
            elif label.startswith("E-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([i_label] * (len(sub_token) - 1) + [label])

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def file_based_convert_examples_to_features(self, input_file, output_file, shard_num, is_training):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._read_tsv(input_file)

        writer = FileWriter(file_name=output_file, shard_num=shard_num)
        nlp_schema = {
            "input_ids": {"type": "int64", "shape": [-1]},
            "input_mask": {"type": "int64", "shape": [-1]},
            "token_type_id": {"type": "int64", "shape": [-1]},
            "label_ids": {"type": "int64", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        data = []
        for index, example in enumerate(examples):
            if index % 1000 == 0:
                print("Writing example %d of %d" % (index, len(examples)))
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer)
            sample = {
                "input_ids": np.array(record.input_ids, dtype=np.int64),
                "input_mask": np.array(record.input_mask, dtype=np.int64),
                "token_type_id": np.array(record.token_type_id, dtype=np.int64),
                "label_ids": np.array([record.label_ids], dtype=np.int64),
            }
            data.append(sample)
        writer.write_raw_data(data)
        writer.commit()

class MRCReader(BaseReader):
    """machine reading comprehension reader"""
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 random_seed=None,
                 for_cn=True,
                 task_id=0,
                 doc_stride=128,
                 max_query_len=64):
        super().__init__(vocab_path,
                         label_map_config,
                         max_seq_len,
                         do_lower_case,
                         random_seed)
        self.for_cn = for_cn
        self.task_id = task_id
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.examples = {}
        self.features = {}

        if random_seed is not None:
            np.random.seed(random_seed)

        self.Example = namedtuple('Example',
                                  ['qas_id', 'question_text', 'doc_tokens', 'orig_answer_text',
                                   'start_position', 'end_position'])
        self.DocSpan = namedtuple("DocSpan", ["start", "length"])

    def _read_json(self, input_file, is_training=True):
        """read json file"""
        examples = []
        with open(input_file, "r", encoding='utf8') as f:
            input_data = json.load(f)["data"]

        def process_one_example(qa, is_training, paragraph_text):
            qas_id = qa["id"]
            question_text = qa["question"]
            start_pos = None
            end_pos = None
            orig_answer_text = None

            if is_training:
                if len(qa["answers"]) != 1:
                    logging.warning(
                        "For training, each question should have exactly 1 answer."
                    )
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                doc_tokens = [
                    paragraph_text[:answer_offset],
                    paragraph_text[answer_offset:answer_offset +
                                   answer_length],
                    paragraph_text[answer_offset + answer_length:]
                ]

                start_pos = 1
                end_pos = 1

                actual_text = " ".join(doc_tokens[start_pos:(end_pos + 1)])
                if actual_text.find(orig_answer_text) == -1:
                    logging.info("Could not find answer: '%s' vs. '%s'",
                                 actual_text, orig_answer_text)
                    return None
            else:
                doc_tokens = tokenize_chinese_chars(paragraph_text)

            example = self.Example(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_pos,
                end_position=end_pos)

            return example

        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    example = process_one_example(qa, is_training, paragraph_text)
                    if example is not None:
                        examples.append(example)

        return examples

    def _improve_answer_span(self, doc_tokens, input_start, input_end,
                             tokenizer, orig_answer_text):
        """improve answer span"""
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """check max context"""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context,
                        num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _convert_example_to_record(self, examples, max_seq_length, tokenizer, is_training):
        records = []
        unique_id = 1000000000
        Record = namedtuple(
            'Record',
            ['input_ids', 'input_mask', 'token_type_id', 'start_position', 'end_position', 'unique_id',
             'example_index', 'doc_span_index', 'tokens', 'token_to_orig_map', 'token_is_max_context'])

        for index, example in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)
            if len(query_tokens) > self.max_query_len:
                query_tokens = query_tokens[0:self.max_query_len]
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.orig_answer_text)

            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(self.DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                token_type_id = []
                tokens.append("[CLS]")
                token_type_id.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    token_type_id.append(0)
                tokens.append("[SEP]")
                token_type_id.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[
                        split_token_index]

                    is_max_context = self._check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    token_type_id.append(1)
                tokens.append("[SEP]")
                token_type_id.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    token_type_id.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(token_type_id) == max_seq_length

                start_position = None
                end_position = None
                if is_training:
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                record = Record(
                    input_ids=input_ids, input_mask=input_mask, token_type_id=token_type_id,
                    start_position=start_position, end_position=end_position, unique_id=unique_id,
                    example_index=index, doc_span_index=doc_span_index, tokens=tokens,
                    token_to_orig_map=token_to_orig_map, token_is_max_context=token_is_max_context,
                    )

                records.append(record)
                unique_id += 1

        return records

    def file_based_convert_examples_to_features(self, input_file, output_file, shard_num, is_training):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._read_json(input_file, is_training)
        writer = FileWriter(file_name=output_file, shard_num=shard_num)
        if is_training:
            nlp_schema = {
                "input_ids": {"type": "int64", "shape": [-1]},
                "input_mask": {"type": "int64", "shape": [-1]},
                "token_type_id": {"type": "int64", "shape": [-1]},
                "start_position": {"type": "int64", "shape": [-1]},
                "end_position": {"type": "int64", "shape": [-1]},
                "unique_id": {"type": "int64", "shape": [-1]},
            }
        else:
            nlp_schema = {
                "input_ids": {"type": "int64", "shape": [-1]},
                "input_mask": {"type": "int64", "shape": [-1]},
                "token_type_id": {"type": "int64", "shape": [-1]},
                "unique_id": {"type": "int64", "shape": [-1]},
            }
        writer.add_schema(nlp_schema, "proprocessed machine reading comprehension dataset")
        data = []
        records = self._convert_example_to_record(examples, self.max_seq_len, self.tokenizer, is_training)
        for index, record in enumerate(records):
            if index % 1000 == 0:
                print(("Writing example %d of %d" % (index, len(records))))
            if is_training:
                sample = {
                    "input_ids": np.array(record.input_ids, dtype=np.int64),
                    "input_mask": np.array(record.input_mask, dtype=np.int64),
                    "token_type_id": np.array(record.token_type_id, dtype=np.int64),
                    "start_position": np.array(record.start_position, dtype=np.int64),
                    "end_position": np.array(record.end_position, dtype=np.int64),
                    "unique_id": np.array(record.unique_id, dtype=np.int64),
                }
            else:
                sample = {
                    "input_ids": np.array(record.input_ids, dtype=np.int64),
                    "input_mask": np.array(record.input_mask, dtype=np.int64),
                    "token_type_id": np.array(record.token_type_id, dtype=np.int64),
                    "unique_id": np.array(record.unique_id, dtype=np.int64),
                }
            data.append(sample)
        writer.write_raw_data(data)
        writer.commit()

    def get_example_features(self, input_file, is_training):
        examples = self._read_json(input_file, is_training)
        records = self._convert_example_to_record(examples, self.max_seq_len, self.tokenizer, is_training)
        return records

    def read_examples(self, input_file, is_training):
        examples = self._read_json(input_file, is_training)
        return examples

reader_dict = {
    'chnsenticorp': ClassifyReader,
    'msra_ner': SequenceLabelingReader,
    'xnli': ClassifyReader,
    'dbqa': ClassifyReader,
    'drcd': MRCReader,
    'cmrc': MRCReader
}

have_label_map = {
    'chnsenticorp': False,
    'msra_ner': True,
    'xnli': True,
    'dbqa': False,
    'drcd': False,
    'cmrc': False
}

def main():
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--task_type", type=str, default="", help="task type to preprocess")
    parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
    parser.add_argument("--label_map_config", type=str, default=None, help="label mapping config file")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--max_query_len", type=int, default=0,
                        help="The maximum total input query length after WordPiece tokenization.")
    parser.add_argument("--do_lower_case", type=str, default="true",
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed number")
    parser.add_argument("--input_file", type=str, default="", help="raw data file")
    parser.add_argument("--output_file", type=str, default="", help="minddata file")
    parser.add_argument("--shard_num", type=int, default=0, help="output file shard number")
    parser.add_argument("--is_training", type=str, default="false",
                        help="Whether the processing dataset is training dataset.")
    args_opt = parser.parse_args()

    if args_opt.max_query_len == 0:
        reader = reader_dict[args_opt.task_type](
            vocab_path=args_opt.vocab_path,
            label_map_config=args_opt.label_map_config if have_label_map[args_opt.task_type] else None,
            max_seq_len=args_opt.max_seq_len,
            do_lower_case=(args_opt.do_lower_case == "true"),
            random_seed=args_opt.random_seed
        )
    else:
        reader = reader_dict[args_opt.task_type](
            vocab_path=args_opt.vocab_path,
            label_map_config=args_opt.label_map_config if have_label_map[args_opt.task_type] else None,
            max_seq_len=args_opt.max_seq_len,
            do_lower_case=(args_opt.do_lower_case == "true"),
            random_seed=args_opt.random_seed,
            max_query_len=args_opt.max_query_len,
        )

    reader.file_based_convert_examples_to_features(input_file=args_opt.input_file,
                                                   output_file=args_opt.output_file,
                                                   shard_num=args_opt.shard_num,
                                                   is_training=(args_opt.is_training == 'true'))

if __name__ == "__main__":
    main()
