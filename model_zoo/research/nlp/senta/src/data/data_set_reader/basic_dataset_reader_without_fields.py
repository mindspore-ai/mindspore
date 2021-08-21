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
"""task reader"""


import os
import json
import logging
from collections import namedtuple
import numpy as np


from src.common.rule import InstanceName
from src.common.register import RegisterSet

from src.data.data_set_reader.basic_dataset_reader import BasicDataSetReader
from src.data.tokenizer.tokenization_spm import preprocess_text
from src.data.tokenizer.tokenization_utils import convert_to_unicode


@RegisterSet.data_set_reader.register
class TaskBaseReader(BasicDataSetReader):
    """task base reader class"""

    def __init__(self,
                 name,
                 fields,
                 config,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 tokenizer="FullTokenizer",
                 text_field_more_than_3=False,
                 use_multi_gpu_test=False):
        BasicDataSetReader.__init__(self, name, fields, config)
        self.text_field_more_than_3 = text_field_more_than_3
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case
        self.vocab_path = vocab_path

        params = {"do_lower_case": do_lower_case}

        tokenizer_class = RegisterSet.tokenizer.__getitem__(tokenizer)
        self.tokenizer = tokenizer_class(vocab_file=vocab_path, params=params)

        self.vocab = self.tokenizer.vocabulary.vocab_dict
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens

        if "train" in self.name:
            self.phase = InstanceName.TRAINING
        elif "dev" in self.name:
            self.phase = InstanceName.EVALUATE
        elif "test" in self.name:
            self.phase = InstanceName.TEST
        else:
            self.phase = None

        self.trainer_id = 0
        self.trainer_nums = 1

        if "train" in self.name or "predict" in self.name:
            self.dev_count = self.trainer_nums
        elif "dev" in self.name or "test" in self.name:
            self.dev_count = 1
            if use_multi_gpu_test:
                self.dev_count = min(self.trainer_nums, 8)
        else:
            logging.info(self.name)
            logging.error("the phase must be train, eval or test !")

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer
        # sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def truncate_seqs(self, tokens_of_sub_sentence, max_num_tokens):
        """truncate_seqs"""
        while True:
            ls = [len(ts) for ts in tokens_of_sub_sentence]
            total_length = sum(ls)
            if total_length <= max_num_tokens:
                break
            max_l = max(ls)
            ind = ls.index(max_l)
            trunc_tokens = tokens_of_sub_sentence[ind]

            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if self.rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def get_all_text_field(self, example):
        """get all text fields"""
        values = []
        for i in range(ord("a"), ord("z")):
            field_name = 'text_' + chr(i)

            if isinstance(example, dict):
                has_field = field_name in example.keys()
            else:
                has_field = field_name in example._fields

            if has_field:
                v = getattr(example, field_name)
                values.append(v)
        return values

    def convert_example_to_record_3(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""
        values = self.get_all_text_field(example)

        all_tokens = []
        for text in values:
            text_a = convert_to_unicode(text)
            tokens = tokenizer.tokenize(text_a)
            all_tokens.append(tokens)

        self.truncate_seqs(all_tokens, max_seq_length - len(all_tokens))

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for i, _tokens in enumerate(all_tokens):
            for token in _tokens:
                tokens.append(token)
                text_type_ids.append(i)
            text_type_ids.append(i)
            tokens.append("[SEP]")

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        task_ids = [0] * len(token_ids)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_id', 'task_ids', 'qid'
             ])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_id=label_id,
            task_ids=task_ids,
            qid=qid)
        return record

    def convert_example_to_record(
            self, example, max_seq_length, tokenizer, is_zh=True):
        """Converts a single `Example` into a single `Record`."""

        if is_zh:
            text_a = convert_to_unicode(example.text_a)
        else:
            text_a = convert_to_unicode(preprocess_text(example.text_a,
                                                        lower=self.do_lower_case))
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if "text_b" in example._fields:
            if is_zh:
                text_b = convert_to_unicode(example.text_b)
            else:
                text_b = convert_to_unicode(preprocess_text(example.text_b,
                                                            lower=self.do_lower_case))
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        task_ids = [0] * len(token_ids)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'task_ids', 'label_id', 'qid'])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            task_ids=task_ids,
            label_id=label_id,
            qid=qid)
        return record

    def read_files(self, data_path, quotechar=None):
        """read file"""
        raise NotImplementedError

    def prepare_batch_data(self, examples, batch_size):
        """generate batch records"""
        batch_records, max_len = [], 0
        for example in examples:
            if "train" in self.name:
                self.current_example += 1

            if self.text_field_more_than_3:
                record = self.convert_example_to_record_3(example, self.max_seq_len,
                                                          self.tokenizer)
            else:
                record = self.convert_example_to_record(example, self.max_seq_len,
                                                        self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self.serialize_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self.serialize_batch_records(batch_records)

    def data_generator(self):
        """generate data"""
        assert os.path.isdir(self.config.data_path), "%s must be a directory that stores data files" \
                                                     % self.config.data_path
        data_files = os.listdir(self.config.data_path)
        epoch = self.config.epoch
        batch_size = self.config.batch_size
        shuffle = self.config.shuffle

        def wrapper():
            """wrapper"""
            all_dev_batches = []
            trainer_id = 0
            for epoch_index in range(epoch):
                if self.phase == InstanceName.TRAINING:
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    self.random_seed = epoch_index
                    self.global_rng = np.random.RandomState(self.random_seed)
                    trainer_id = self.trainer_id
                else:
                    self.random_seed = 0
                    self.global_rng = np.random.RandomState(self.random_seed)
                    trainer_id = self.trainer_id

                for input_file in data_files:
                    current_examples = self.read_files(
                        os.path.join(self.config.data_path, input_file))

                    if shuffle:
                        self.global_rng.shuffle(current_examples)
                    for batch_data in self.prepare_batch_data(
                            current_examples, batch_size):
                        if len(all_dev_batches) < self.dev_count:
                            all_dev_batches.append(batch_data)
                        if len(all_dev_batches) == self.dev_count:
                            # trick: handle batch inconsistency caused by data
                            # sharding for each trainer
                            yield all_dev_batches[trainer_id]
                            all_dev_batches = []
                    if self.phase != InstanceName.TRAINING:
                        if trainer_id < len(all_dev_batches):
                            yield all_dev_batches[trainer_id]

        return wrapper

    def serialize_batch_records(self, batch_records):
        """serialize_batch_records"""
        raise NotImplementedError
