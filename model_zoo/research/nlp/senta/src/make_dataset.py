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
"""import"""
import os
from src.common import register
from src.data.data_set import DataSet
from src.utils import args
from src.utils import params
from src.common.rule import InstanceName
from src.config import sstcfg, semcfg

from mindspore.mindrecord import FileWriter


def dataset_reader_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    reader = DataSet(params_dict)
    reader.build()

    return reader


def make_dataset(wrapper=None, output_path=None, task_name=None, mode="train"):
    '''
    :param wrapper:
    :param output_path:
    :param task_name:
    :param mode:
    :return:
    '''
    MINDRECORD_FILE_PATH = os.path.join(output_path, task_name, \
        task_name + "_" + mode + ".mindrecord")
    if not os.path.exists(output_path + task_name):
        os.makedirs(output_path + task_name)
    if os.path.exists(MINDRECORD_FILE_PATH):
        os.remove(MINDRECORD_FILE_PATH)
        os.remove(MINDRECORD_FILE_PATH + ".db")
    writer = FileWriter(file_name=MINDRECORD_FILE_PATH, shard_num=1)
    datalist = []
    if task_name == 'Sem-L':
        nlp_schema = {
            InstanceName.SRC_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.SENTENCE_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.POS_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.MASK_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.LABEL: {"type": "int32", "shape": [-1]},
            InstanceName.RECORD_ID: {"type": "int32", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        for i in wrapper():
            data = {
                InstanceName.SRC_IDS: i[0],
                InstanceName.SENTENCE_IDS: i[1],
                InstanceName.POS_IDS: i[2],
                InstanceName.MASK_IDS: i[4],
                InstanceName.LABEL: i[5],
                InstanceName.RECORD_ID: i[6]
            }
            datalist.append(data)
    if task_name == 'SST-2':
        nlp_schema = {
            InstanceName.RECORD_ID: {"type": "int32", "shape": [-1]},
            InstanceName.LABEL: {"type": "int32", "shape": [-1]},
            InstanceName.SRC_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.SENTENCE_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.POS_IDS: {"type": "int32", "shape": [-1]},
            InstanceName.MASK_IDS: {"type": "int32", "shape": [-1]}
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        for i in wrapper():
            data = {
                InstanceName.RECORD_ID: i[0],
                InstanceName.LABEL: i[1],
                InstanceName.SRC_IDS: i[2],
                InstanceName.SENTENCE_IDS: i[3],
                InstanceName.POS_IDS: i[4],
                InstanceName.MASK_IDS: i[5]
            }
            datalist.append(data)

    writer.write_raw_data(datalist)
    writer.commit()
    print("write success")


if __name__ == "__main__":
    args = args.build_common_arguments()
    if args.job == "SST-2":
        param_dict = sstcfg
        _params = params.replace_none(param_dict)
        register.import_modules()
        dataset_reader_params_dict = _params.get("dataset_reader")
        dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)
        train_wrapper = dataset_reader.train_reader.data_generator()
        make_dataset(
            wrapper=train_wrapper,
            output_path='../data/',
            task_name=args.job,
            mode="train")
        dev_wrapper = dataset_reader.dev_reader.data_generator()
        make_dataset(
            wrapper=dev_wrapper,
            output_path='../data/',
            task_name=args.job,
            mode="dev")
    if args.job == "Sem-L":
        param_dict = semcfg
        _params = params.replace_none(param_dict)
        register.import_modules()
        dataset_reader_params_dict = _params.get("dataset_reader")
        dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)
        train_wrapper = dataset_reader.train_reader.data_generator()
        make_dataset(
            wrapper=train_wrapper,
            output_path='../data/',
            task_name=args.job,
            mode="train")
        dev_wrapper = dataset_reader.test_reader.data_generator()
        make_dataset(
            wrapper=dev_wrapper,
            output_path='../data/',
            task_name=args.job,
            mode="dev")
