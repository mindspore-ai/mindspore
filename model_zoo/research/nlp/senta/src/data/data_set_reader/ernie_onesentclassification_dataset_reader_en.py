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
from src.data.data_set_reader.basic_dataset_reader import BasicDataSetReader


@RegisterSet.data_set_reader.register
class OneSentClassifyReaderEn(BasicDataSetReader):
    """BasicDataSetReader:
    """

    def __init__(self, name, fields, config):
        BasicDataSetReader.__init__(self, name, fields, config)

        self.trainer_id = 0
        self.trainer_nums = 1

        if "train" in self.name or "predict" in self.name:
            self.dev_count = self.trainer_nums
        elif "dev" in self.name or "test" in self.name:
            self.dev_count = 1
            use_multi_gpu_test = True
            if use_multi_gpu_test:
                self.dev_count = min(self.trainer_nums, 8)
        else:
            logging.error("the phase must be train, eval or test !")

    def read_files(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            try:
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                headers = next(reader)
                text_indices = [
                    index for index, h in enumerate(headers) if h != "label"
                ]

                Example = namedtuple('Example', headers)

                examples = []
                # i = 0
                for line in reader:
                    for index, text in enumerate(line):
                        if index in text_indices:
                            line[index] = text  # .replace(' ', '')
                    example = Example(*line)
                    examples.append(example)
                return examples
            except IOError:
                logging.error("error in read tsv")

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
            all_dev_batches = []
            for epoch_index in range(self.config.epoch):
                self.current_example = 0
                self.current_epoch = epoch_index
                self.global_rng = np.random.RandomState(epoch_index)

                for input_file in data_files:
                    examples = self.read_files(os.path.join(
                        self.config.data_path, input_file))
                    if self.config.shuffle:
                        self.global_rng.shuffle(examples)

                    for batch_data in self.prepare_batch_data(
                            examples, self.config.batch_size):
                        if len(all_dev_batches) < self.dev_count:
                            all_dev_batches.append(batch_data)
                        if len(all_dev_batches) == self.dev_count:
                            # trick: handle batch inconsistency caused by data
                            # sharding for each trainer
                            yield all_dev_batches[self.trainer_id]
                            all_dev_batches = []
                    if "train" not in self.name:
                        if self.trainer_id < len(all_dev_batches):
                            yield all_dev_batches[self.trainer_id]

        return wrapper

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
