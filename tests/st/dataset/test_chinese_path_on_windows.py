# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This is the test module for mindrecord
"""
import os
import platform
from io import BytesIO
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore.mindrecord import FileWriter, SUCCESS

FILES_NUM = 4
CV_MINDRECORD_FILE = "../data/test.mindrecord"
CV_DIR_NAME_CN = "../data/数据集/train/"
FILE_NAME = "test.mindrecord"
FILE_NAME2 = "./训练集/test.mindrecord"

def add_and_remove_cv_file(mindrecord):
    """add/remove cv file"""
    try:
        if os.path.exists("{}".format(mindrecord)):
            os.remove("{}".format(mindrecord))
        if os.path.exists("{}.db".format(mindrecord)):
            os.remove("{}.db".format(mindrecord))
    except Exception as error:
        raise error

def write_read_mindrecord(mindrecord):
    writer = FileWriter(file_name=mindrecord, shard_num=1)

    cv_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema, "it is a cv dataset")

    writer.add_index(["file_name", "label"])

    data = []
    for i in range(100):
        i += 1

        sample = {}
        white_io = BytesIO()
        Image.new('RGB', (i*10, i*10), (255, 255, 255)).save(white_io, 'JPEG')
        sample['file_name'] = str(i) + ".jpg"
        sample['label'] = i
        sample['data'] = white_io.getvalue()

        data.append(sample)
        if i % 10 == 0:
            writer.write_raw_data(data)
            data = []

    if data:
        writer.write_raw_data(data)

    assert writer.commit() == SUCCESS

    if not os.path.exists(mindrecord):
        raise "generator mindrecord file failed"
    if not os.path.exists(mindrecord + ".db"):
        raise "generator mindrecord db file failed"

    data_set = ds.MindDataset(dataset_files=mindrecord)
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
    count = 0
    for _ in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    assert count == 100

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_chinese_path_on_windows():
    """
    Feature: test chinese path on windows platform
    Description: None
    Expectation: raise axception
    """

    if platform.system().lower() != "windows":
        pass

    current_pwd = os.getcwd()

    # current dir in english, mindrecord path in english
    dir_path = "./"
    mindrecord_path = CV_MINDRECORD_FILE

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in english, mindrecord path in chinese
    dir_path = "./"
    mindrecord_path = CV_DIR_NAME_CN + "/" + FILE_NAME

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in chinese, mindrecord path in english
    dir_path = CV_DIR_NAME_CN
    mindrecord_path = FILE_NAME

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in chinese, mindrecord path in chinese
    dir_path = CV_DIR_NAME_CN
    mindrecord_path = FILE_NAME2

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)


if __name__ == '__main__':
    test_chinese_path_on_windows()
