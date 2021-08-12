"""task reader"""
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
import csv
import json
import logging
from collections import namedtuple
import numpy as np


from src.common.register import RegisterSet
from src.common.rule import InstanceName
from src.data.data_set_reader.basic_dataset_reader import BasicDataSetReader
from src.data.data_set_reader.basic_dataset_reader_without_fields import TaskBaseReader

from src.data.util_helper import pad_batch_data

@RegisterSet.data_set_reader.register
class RobertaTwoSentClassifyReaderEn(TaskBaseReader):
    """classify reader"""
    def __init__(self, name, fields, config):

        BasicDataSetReader.__init__(self, name, fields, config)
        self.max_seq_len = config.extra_params.get("max_seq_len", 512)
        self.vocab_path = config.extra_params.get("vocab_path")
        self.text_field_more_than_3 = config.extra_params.get("text_field_more_than_3", False)
        self.data_augmentation = config.extra_params.get("data_augmentation", False)
        self.in_tokens = config.extra_params.get("in_tokens", False)
        self.tokenizer_name = config.extra_params.get("tokenizer", "GptBpeTokenizer")
        self.is_classify = config.extra_params.get("is_classify", True)
        self.is_regression = config.extra_params.get("is_regression", False)
        self.use_multi_gpu_test = config.extra_params.get("use_multi_gpu_test", True)
        self.label_map_config = config.extra_params.get("label_map_config")
        self.bpe_json_file = config.extra_params.get("bpe_json_file", False)
        self.bpe_vocab_file = config.extra_params.get("bpe_vocab_file", False)

        params = {"bpe_json_file": self.bpe_json_file, "bpe_vocab_file": self.bpe_vocab_file}
        tokenizer_class = RegisterSet.tokenizer.__getitem__(self.tokenizer_name)
        self.tokenizer = tokenizer_class(vocab_file=self.vocab_path, params=params)

        self.vocab = self.tokenizer.vocabulary.vocab_dict
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

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


        if "train" in self.name:
            self.dev_count = self.trainer_nums
        elif "dev" in self.name or "test" in self.name or "predict" in self.name:
            self.dev_count = 1
            if self.use_multi_gpu_test:
                self.dev_count = min(self.trainer_nums, 8)
        else:
            logging.error("the phase must be train, eval or test !")

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if self.label_map_config:
            with open(self.label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None


    def read_files(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        f = open(input_file, "r")
        try:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            label_indices = [
                index for index, h in enumerate(headers) if h == "label"
            ]

            Example = namedtuple('Example', headers)

            examples = []
            # i = 0
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text #.replace(' ', '')
                    elif index in label_indices:
                        text_ind = text_indices[0]
                        text = line[text_ind]

                example = Example(*line)
                examples.append(example)
            return examples
        except IOError:
            logging.error("error in read tsv")

    def serialize_batch_records(self, batch_records):
        """pad batch records"""
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_task_ids = [record.task_ids for record in batch_records]
        if "predict" not in self.name:
            batch_labels = [record.label_id for record in batch_records]
            if self.is_classify:
                batch_labels = np.array(batch_labels).astype("int32").reshape([-1, 1])
            elif self.is_regression:
                batch_labels = np.array(batch_labels).astype("float32").reshape([-1, 1])
        else:
            if self.is_classify:
                batch_labels = np.array([]).astype("int32").reshape([-1, 1])
            elif self.is_regression:
                batch_labels = np.array([]).astype("float32").reshape([-1, 1])

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int32").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int32").reshape([-1, 1])

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_task_ids = pad_batch_data(
            batch_task_ids, pad_idx=0)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, batch_labels, batch_qids
        ]

        return return_list
