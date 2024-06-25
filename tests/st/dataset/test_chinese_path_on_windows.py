# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.mindrecord import FileWriter
from tests.mark_utils import arg_mark

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

    writer.commit()

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


@arg_mark(plat_marks=['cpu_windows'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_chinese_path_on_windows():
    """
    Feature: test chinese path on windows platform
    Description: None
    Expectation: raise axception
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    cv_mindrecord_file = "./data/" + mindrecord_file_name
    cv_dir_name_cn = "./data/数据集/train/"
    file_name = mindrecord_file_name
    file_name2 = "./训练集/" + mindrecord_file_name

    if platform.system().lower() != "windows":
        return

    current_pwd = os.getcwd()

    # create chinese path for test
    if not os.path.exists("data/数据集/train/训练集"):
        os.makedirs("data/数据集/train/训练集")

    # current dir in english, mindrecord path in english
    dir_path = "./"
    mindrecord_path = cv_mindrecord_file

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in english, mindrecord path in chinese
    dir_path = "./"
    mindrecord_path = cv_dir_name_cn + file_name

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in chinese, mindrecord path in english
    dir_path = cv_dir_name_cn
    mindrecord_path = file_name

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in chinese, mindrecord path in chinese
    dir_path = cv_dir_name_cn
    mindrecord_path = file_name2

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)


@arg_mark(plat_marks=['cpu_windows'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_backslash_path_on_windows():
    """
    Feature: test path on windows platform which contains both slash and backslash
    Description: None
    Expectation: raise axception
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    cv_dir_name_cn = "./data/数据集/train/"
    file_name2 = '.\\训练集\\' + mindrecord_file_name

    if platform.system().lower() != "windows":
        return

    current_pwd = os.getcwd()

    # create chinese path for test
    if not os.path.exists(os.path.join(cv_dir_name_cn, "训练集")):
        os.makedirs(os.path.join(cv_dir_name_cn, "训练集"))
    if not os.path.exists(os.path.join(cv_dir_name_cn, "data")):
        os.makedirs(os.path.join(cv_dir_name_cn, "data"))

    # current dir in chinese, mindrecord path in chinese with back slash
    dir_path = cv_dir_name_cn
    mindrecord_path = file_name2

    add_and_remove_cv_file(dir_path + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord(mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + mindrecord_path)

    # current dir in chinese, mindrecord path in english with back slash
    dir_path = "./data/数据集/train"
    mindrecord_path = mindrecord_file_name

    add_and_remove_cv_file(dir_path + '/' + mindrecord_path)

    os.chdir(dir_path)
    write_read_mindrecord('.\\' + mindrecord_path)

    os.chdir(current_pwd)
    add_and_remove_cv_file(dir_path + '/' + mindrecord_path)

    # current dir in chinese, mindrecord path in english with back slash
    dir_path = "./data/数据集/train"
    mindrecord_path = mindrecord_file_name

    add_and_remove_cv_file(dir_path + '/' + mindrecord_path)

    write_read_mindrecord(dir_path + '\\' + mindrecord_path)

    add_and_remove_cv_file(dir_path + '/' + mindrecord_path)

    # current dir in chinese, mindrecord path in english with back slash
    dir_path = "./data/数据集/train"
    mindrecord_path = 'data/' + mindrecord_file_name

    add_and_remove_cv_file(dir_path + '/' + mindrecord_path)

    write_read_mindrecord(dir_path + '\\' + mindrecord_path)

    add_and_remove_cv_file(dir_path + '/' + mindrecord_path)


if __name__ == '__main__':
    test_chinese_path_on_windows()
    test_backslash_path_on_windows()
