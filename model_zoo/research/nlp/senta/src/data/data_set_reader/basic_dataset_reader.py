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
"""
:py:class:`BasicDataSetReader`
"""
import csv
import os
import logging
from collections import namedtuple
import numpy as np

from src.common.register import RegisterSet


@RegisterSet.data_set_reader.register
class BasicDataSetReader():
    """BasicDataSetReader:
    """

    def __init__(self, name, fields, config):
        self.name = name
        self.fields = fields
        self.config = config
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

    def data_generator(self):
        """
        :return:
        """
        assert os.path.isdir(self.config.data_path), "%s must be a directory that stores data files" \
                                                     % self.config.data_path
        data_files = os.listdir(self.config.data_path)

        def wrapper():
            """
            :return:
            """
            for epoch_index in range(self.config.epoch):
                self.current_example = 0
                self.current_epoch = epoch_index

                for input_file in data_files:
                    examples = self.read_files(os.path.join(
                        self.config.data_path, input_file))
                    if self.config.shuffle:
                        np.random.shuffle(examples)

                    for batch_data in self.prepare_batch_data(
                            examples, self.config.batch_size):
                        yield batch_data

        return wrapper

    def read_files(self, file_path, quotechar=None):
        """
        :param file_path
        :return:
        """
        with open(file_path, "r") as f:
            try:
                examples = []
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                len_fields = len(self.fields)
                field_names = []
                for filed in self.fields:
                    field_names.append(filed.name)

                self.Example = namedtuple('Example', field_names)

                for line in reader:
                    if len(line) == len_fields:
                        example = self.Example(*line)
                        examples.append(example)
                return examples

            except IOError:
                logging.error("error in read tsv")

    def prepare_batch_data(self, examples, batch_size):
        """
        prepare_batch_data
        """
        batch_records = []
        for example in examples:
            self.current_example += 1
            if len(batch_records) < batch_size:
                batch_records.append(example)
            else:
                yield self.serialize_batch_records(batch_records)
                batch_records = [example]

        if batch_records:
            yield self.serialize_batch_records(batch_records)

    def serialize_batch_records(self, batch_records):
        """
        :param batch_records:
        :return:
        """
        return_list = []
        example = batch_records[0]
        for index in range(len(example._fields)):
            text_batch = []
            for record in batch_records:
                text_batch.append(record[index])

            id_list = self.fields[index].field_reader.convert_texts_to_ids(
                text_batch)
            return_list.extend(id_list)

        return return_list
