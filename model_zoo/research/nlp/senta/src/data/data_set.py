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
:py:class:`DataSet`
"""
from src.common.register import RegisterSet
from src.data.field import Field
from src.data.reader_config import ReaderConfig


class DataSet():
    """DataSet"""

    def __init__(self, params_dict):
        """"""
        self.train_reader = None
        self.test_reader = None
        self.dev_reader = None
        self.predict_reader = None

        self.params_dict = params_dict

    def build(self):
        """
        :return:
        """
        reader_list = []
        data_set_reader_dict = {}

        if self.params_dict.__contains__("train_reader"):
            reader_list.append("train_reader")
        if self.params_dict.__contains__("test_reader"):
            reader_list.append("test_reader")
        if self.params_dict.__contains__("dev_reader"):
            reader_list.append("dev_reader")
        if self.params_dict.__contains__("predict_reader"):
            reader_list.append("predict_reader")

        for reader_name in reader_list:
            cfg_list = self.params_dict.get(reader_name).get("fields")
            train_fields = []

            for item in cfg_list:
                item_field = Field()
                item_field.build(item)
                if item_field.reader_info and item_field.reader_info.get(
                        "type", None):
                    reader_class = RegisterSet.field_reader.__getitem__(
                        item_field.reader_info["type"])
                    field_reader = reader_class(item_field)
                    item_field.field_reader = field_reader
                    train_fields.append(item_field)

            reader_cfg = ReaderConfig()
            reader_cfg.build(self.params_dict.get(reader_name).get("config"))

            dataset_reader_name = self.params_dict.get(reader_name).get("type")
            dataset_reader_class = RegisterSet.data_set_reader.__getitem__(
                dataset_reader_name)
            one_reader = dataset_reader_class(
                name=reader_name, fields=train_fields, config=reader_cfg)

            data_set_reader_dict[reader_name] = one_reader

        if data_set_reader_dict.__contains__("train_reader"):
            self.train_reader = data_set_reader_dict["train_reader"]

        if data_set_reader_dict.__contains__("test_reader"):
            self.test_reader = data_set_reader_dict["test_reader"]

        if data_set_reader_dict.__contains__("dev_reader"):
            self.dev_reader = data_set_reader_dict["dev_reader"]

        if data_set_reader_dict.__contains__("predict_reader"):
            self.predict_reader = data_set_reader_dict["predict_reader"]
        elif self.train_reader:
            cfg_list = self.params_dict.get("train_reader").get("fields")
            predict_fields = []
            for item in cfg_list:
                item_field = Field()
                item_field.build(item)
                if item_field.reader_info and item_field.reader_info.get(
                        "type", None):
                    reader_class = RegisterSet.field_reader.__getitem__(
                        item_field.reader_info["type"])
                    field_reader = reader_class(item_field)
                    item_field.field_reader = field_reader
                    predict_fields.append(item_field)

            reader_cfg = ReaderConfig()
            reader_cfg.build(self.params_dict.get(
                "train_reader").get("config"))

            dataset_reader_name = self.params_dict.get(
                "train_reader").get("type")
            dataset_reader_class = RegisterSet.data_set_reader.__getitem__(
                dataset_reader_name)
            self.predict_reader = dataset_reader_class(name="predict_reader", fields=predict_fields,
                                                       config=reader_cfg)
