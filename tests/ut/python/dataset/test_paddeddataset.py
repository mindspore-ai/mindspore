# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from io import BytesIO
import copy
import os
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision as V_C

FILES_NUM = 4
CV_DIR_NAME = "../data/mindrecord/testImageNetData"


def generator_5():
    for i in range(0, 5):
        yield (np.array([i]),)


def generator_8():
    for i in range(5, 8):
        yield (np.array([i]),)


def generator_10():
    for i in range(0, 10):
        yield (np.array([i]),)


def generator_20():
    for i in range(10, 20):
        yield (np.array([i]),)


def generator_30():
    for i in range(20, 30):
        yield (np.array([i]),)


def test_TFRecord_Padded():
    """
    Feature: PaddedDataset
    Description: Test padding PaddedDataset on TFRecordDataset
    Expectation: Output is equal to the expected output
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_dir = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
    result_list = [[159109, 2], [192607, 3], [179251, 4], [1, 5]]
    verify_list = []
    shard_num = 4
    for i in range(shard_num):
        data = ds.TFRecordDataset(data_dir, schema_dir, columns_list=["image"],
                                  shuffle=False, shard_equal_rows=True)

        padded_samples = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)},
                          {'image': np.zeros(3, np.uint8)}, {'image': np.zeros(4, np.uint8)},
                          {'image': np.zeros(5, np.uint8)}]

        padded_ds = ds.PaddedDataset(padded_samples)
        concat_ds = data + padded_ds
        testsampler = ds.DistributedSampler(num_shards=shard_num, shard_id=i, shuffle=False, num_samples=None)
        concat_ds.use_sampler(testsampler)
        shard_list = []
        for item in concat_ds.create_dict_iterator(num_epochs=1, output_numpy=True):
            shard_list.append(len(item['image']))
        verify_list.append(shard_list)
    assert verify_list == result_list


def test_GeneratorDataSet_Padded():
    """
    Feature: PaddedDataset
    Description: Test padding GeneratorDataset with another GeneratorDataset
    Expectation: Output is equal to the expected output
    """
    result_list = []
    for i in range(10):
        tem_list = []
        tem_list.append(i)
        tem_list.append(10 + i)
        result_list.append(tem_list)

    verify_list = []
    data1 = ds.GeneratorDataset(generator_20, ["col1"])
    data2 = ds.GeneratorDataset(generator_10, ["col1"])
    data3 = data2 + data1
    shard_num = 10
    for i in range(shard_num):
        distributed_sampler = ds.DistributedSampler(num_shards=shard_num, shard_id=i, shuffle=False, num_samples=None)
        data3.use_sampler(distributed_sampler)
        tem_list = []
        for ele in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            tem_list.append(ele['col1'][0])
        verify_list.append(tem_list)

    assert verify_list == result_list


def test_Reapeat_afterPadded():
    """
    Feature: PaddedDataset
    Description: Test padding PaddedDataset with another PaddedDataset
    Expectation: Output is equal to the expected output
    """
    result_list = [1, 3, 5, 7]
    verify_list = []

    data1 = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)},
             {'image': np.zeros(3, np.uint8)}, {'image': np.zeros(4, np.uint8)},
             {'image': np.zeros(5, np.uint8)}]
    data2 = [{'image': np.zeros(6, np.uint8)}, {'image': np.zeros(7, np.uint8)},
             {'image': np.zeros(8, np.uint8)}]

    ds1 = ds.PaddedDataset(data1)
    ds2 = ds.PaddedDataset(data2)
    ds3 = ds1 + ds2

    testsampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=None)
    ds3.use_sampler(testsampler)
    repeat_num = 2
    ds3 = ds3.repeat(repeat_num)
    for item in ds3.create_dict_iterator(num_epochs=1, output_numpy=True):
        verify_list.append(len(item['image']))

    assert verify_list == result_list * repeat_num


def test_bath_afterPadded():
    """
    Feature: PaddedDataset
    Description: Test padding PaddedDataset with another PaddedDataset followed by batch op
    Expectation: Output is equal to the expected output
    """
    data1 = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(1, np.uint8)},
             {'image': np.zeros(1, np.uint8)}, {'image': np.zeros(1, np.uint8)},
             {'image': np.zeros(1, np.uint8)}]
    data2 = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(1, np.uint8)},
             {'image': np.zeros(1, np.uint8)}]

    ds1 = ds.PaddedDataset(data1)
    ds2 = ds.PaddedDataset(data2)
    ds3 = ds1 + ds2

    testsampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=None)
    ds3.use_sampler(testsampler)

    ds4 = ds3.batch(2)
    assert sum([1 for _ in ds4]) == 2


def test_Unevenly_distributed():
    """
    Feature: PaddedDataset
    Description: Test padding PaddedDataset with another PaddedDataset that is unevenly distributed
    Expectation: Output is equal to the expected output
    """
    result_list = [[1, 4, 7], [2, 5, 8], [3, 6]]
    verify_list = []

    data1 = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)},
             {'image': np.zeros(3, np.uint8)}, {'image': np.zeros(4, np.uint8)},
             {'image': np.zeros(5, np.uint8)}]
    data2 = [{'image': np.zeros(6, np.uint8)}, {'image': np.zeros(7, np.uint8)},
             {'image': np.zeros(8, np.uint8)}]

    testsampler = ds.DistributedSampler(num_shards=4, shard_id=0, shuffle=False, num_samples=None, offset=1)

    ds1 = ds.PaddedDataset(data1)
    ds2 = ds.PaddedDataset(data2)
    ds3 = ds1 + ds2
    numShard = 3
    for i in range(numShard):
        tem_list = []
        testsampler = ds.DistributedSampler(num_shards=numShard, shard_id=i, shuffle=False, num_samples=None)
        ds3.use_sampler(testsampler)
        for item in ds3.create_dict_iterator(num_epochs=1, output_numpy=True):
            tem_list.append(len(item['image']))
        verify_list.append(tem_list)
    assert verify_list == result_list


def test_three_datasets_connected():
    """
    Feature: PaddedDataset
    Description: Test padding 3 connected GeneratorDatasets
    Expectation: Output is equal to the expected output
    """
    result_list = []
    for i in range(10):
        tem_list = []
        tem_list.append(i)
        tem_list.append(10 + i)
        tem_list.append(20 + i)
        result_list.append(tem_list)

    verify_list = []
    data1 = ds.GeneratorDataset(generator_10, ["col1"])
    data2 = ds.GeneratorDataset(generator_20, ["col1"])
    data3 = ds.GeneratorDataset(generator_30, ["col1"])
    data4 = data1 + data2 + data3
    shard_num = 10
    for i in range(shard_num):
        distributed_sampler = ds.DistributedSampler(num_shards=shard_num, shard_id=i, shuffle=False, num_samples=None)
        data4.use_sampler(distributed_sampler)
        tem_list = []
        for ele in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
            tem_list.append(ele['col1'][0])
        verify_list.append(tem_list)

    assert verify_list == result_list


def test_raise_error():
    """
    Feature: PaddedDataset
    Description: Test padding a PaddedDataset after a batch op with a PaddedDataset, then apply sampler op
    Expectation: Correct error is raised as expected
    """
    data1 = [{'image': np.zeros(0, np.uint8)}, {'image': np.zeros(0, np.uint8)},
             {'image': np.zeros(0, np.uint8)}, {'image': np.zeros(0, np.uint8)},
             {'image': np.zeros(0, np.uint8)}]
    data2 = [{'image': np.zeros(0, np.uint8)}, {'image': np.zeros(0, np.uint8)},
             {'image': np.zeros(0, np.uint8)}]

    ds1 = ds.PaddedDataset(data1)
    ds4 = ds1.batch(2)
    ds2 = ds.PaddedDataset(data2)
    ds3 = ds4 + ds2

    with pytest.raises(TypeError) as excinfo:
        testsampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=None)
        ds3.use_sampler(testsampler)
        assert excinfo.type == 'TypeError'

    with pytest.raises(TypeError) as excinfo:
        otherSampler = ds.SequentialSampler()
        ds3.use_sampler(otherSampler)
        assert excinfo.type == 'TypeError'

    with pytest.raises(ValueError) as excinfo:
        testsampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=True, num_samples=None)
        ds3.use_sampler(testsampler)
        assert excinfo.type == 'ValueError'

    with pytest.raises(ValueError) as excinfo:
        testsampler = ds.DistributedSampler(num_shards=2, shard_id=0, shuffle=False, num_samples=5)
        ds3.use_sampler(testsampler)
        assert excinfo.type == 'ValueError'

def test_imagefolder_error():
    """
    Feature: PaddedDataset
    Description: Test padding an ImageFolderDataset with num_samples with PaddedDataset
    Expectation: Error is raised as expected
    """
    data_dir = "../data/dataset/testPK/data"
    data = ds.ImageFolderDataset(data_dir, num_samples=14)

    data1 = [{'image': np.zeros(1, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(2, np.uint8), 'label': np.array(1, np.int32)},
             {'image': np.zeros(3, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(4, np.uint8), 'label': np.array(1, np.int32)},
             {'image': np.zeros(5, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(6, np.uint8), 'label': np.array(1, np.int32)}]

    data2 = ds.PaddedDataset(data1)
    data3 = data + data2
    with pytest.raises(ValueError) as excinfo:
        testsampler = ds.DistributedSampler(num_shards=5, shard_id=4, shuffle=False, num_samples=None)
        data3.use_sampler(testsampler)
        assert excinfo.type == 'ValueError'

def test_imagefolder_padded():
    """
    Feature: PaddedDataset
    Description: Test padding an ImageFolderDataset without num_samples with PaddedDataset
    Expectation: Output is equal to the expected output
    """
    data_dir = "../data/dataset/testPK/data"
    data = ds.ImageFolderDataset(data_dir)

    data1 = [{'image': np.zeros(1, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(2, np.uint8), 'label': np.array(1, np.int32)},
             {'image': np.zeros(3, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(4, np.uint8), 'label': np.array(1, np.int32)},
             {'image': np.zeros(5, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(6, np.uint8), 'label': np.array(1, np.int32)}]

    data2 = ds.PaddedDataset(data1)
    data3 = data + data2
    testsampler = ds.DistributedSampler(num_shards=5, shard_id=4, shuffle=False, num_samples=None)
    data3.use_sampler(testsampler)
    assert sum([1 for _ in data3]) == 10
    verify_list = []

    for ele in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        verify_list.append(len(ele['image']))
    assert verify_list[8] == 1
    assert verify_list[9] == 6


def test_imagefolder_padded_with_decode():
    """
    Feature: PaddedDataset
    Description: Test padding an ImageFolderDataset with PaddedDataset followed by a Decode op
    Expectation: Output is equal to the expected output
    """
    num_shards = 5
    count = 0
    for shard_id in range(num_shards):
        data_dir = "../data/dataset/testPK/data"
        data = ds.ImageFolderDataset(data_dir)

        white_io = BytesIO()
        Image.new('RGB', (224, 224), (255, 255, 255)).save(white_io, 'JPEG')
        padded_sample = {}
        padded_sample['image'] = np.array(bytearray(white_io.getvalue()), dtype='uint8')
        padded_sample['label'] = np.array(-1, np.int32)

        white_samples = [padded_sample, padded_sample, padded_sample, padded_sample]
        data2 = ds.PaddedDataset(white_samples)
        data3 = data + data2

        testsampler = ds.DistributedSampler(num_shards=num_shards, shard_id=shard_id, shuffle=False, num_samples=None)
        data3.use_sampler(testsampler)
        data3 = data3.map(operations=V_C.Decode(), input_columns="image")
        shard_sample_count = 0
        for ele in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            print("label: {}".format(ele['label']))
            count += 1
            shard_sample_count += 1
        assert shard_sample_count in (9, 10)
    assert count == 48


def test_imagefolder_padded_with_decode_and_get_dataset_size():
    """
    Feature: PaddedDataset
    Description: Test padding an ImageFolderDataset with PaddedDataset followed by get_dataset_size and a Decode op
    Expectation: Output is equal to the expected output
    """
    num_shards = 5
    count = 0
    for shard_id in range(num_shards):
        data_dir = "../data/dataset/testPK/data"
        data = ds.ImageFolderDataset(data_dir)

        white_io = BytesIO()
        Image.new('RGB', (224, 224), (255, 255, 255)).save(white_io, 'JPEG')
        padded_sample = {}
        padded_sample['image'] = np.array(bytearray(white_io.getvalue()), dtype='uint8')
        padded_sample['label'] = np.array(-1, np.int32)

        white_samples = [padded_sample, padded_sample, padded_sample, padded_sample]
        data2 = ds.PaddedDataset(white_samples)
        data3 = data + data2

        testsampler = ds.DistributedSampler(num_shards=num_shards, shard_id=shard_id, shuffle=False, num_samples=None)
        data3.use_sampler(testsampler)
        shard_dataset_size = data3.get_dataset_size()
        data3 = data3.map(operations=V_C.Decode(), input_columns="image")
        shard_sample_count = 0
        for ele in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            print("label: {}".format(ele['label']))
            count += 1
            shard_sample_count += 1
        assert shard_sample_count in (9, 10)
        assert shard_dataset_size == shard_sample_count
    assert count == 48


def test_more_shard_padded():
    """
    Feature: PaddedDataset
    Description: Test padding GeneratorDataset with another GeneratorDataset and
        PaddedDataset with another PaddedDataset with larger num_shards used
    Expectation: Output is equal to the expected output
    """
    result_list = []
    for i in range(8):
        result_list.append(1)
    result_list.append(0)

    data1 = ds.GeneratorDataset(generator_5, ["col1"])
    data2 = ds.GeneratorDataset(generator_8, ["col1"])
    data3 = data1 + data2
    vertifyList = []
    numShard = 9
    for i in range(numShard):
        tem_list = []
        testsampler = ds.DistributedSampler(num_shards=numShard, shard_id=i, shuffle=False, num_samples=None)
        data3.use_sampler(testsampler)
        for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            tem_list.append(item['col1'])
        vertifyList.append(tem_list)

    assert [len(ele) for ele in vertifyList] == result_list

    vertifyList1 = []
    result_list1 = []
    for i in range(8):
        result_list1.append([i + 1])
    result_list1.append([])

    data1 = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)},
             {'image': np.zeros(3, np.uint8)}, {'image': np.zeros(4, np.uint8)},
             {'image': np.zeros(5, np.uint8)}]
    data2 = [{'image': np.zeros(6, np.uint8)}, {'image': np.zeros(7, np.uint8)},
             {'image': np.zeros(8, np.uint8)}]

    ds1 = ds.PaddedDataset(data1)
    ds2 = ds.PaddedDataset(data2)
    ds3 = ds1 + ds2

    for i in range(numShard):
        tem_list = []
        testsampler = ds.DistributedSampler(num_shards=numShard, shard_id=i, shuffle=False, num_samples=None)
        ds3.use_sampler(testsampler)
        for item in ds3.create_dict_iterator(num_epochs=1, output_numpy=True):
            tem_list.append(len(item['image']))
        vertifyList1.append(tem_list)

    assert vertifyList1 == result_list1


def get_data(dir_name):
    """
    usage: get data from imagenet dataset

    params:
    dir_name: directory containing folder images and annotation information
    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "images")
    ann_file = os.path.join(dir_name, "annotation.txt")
    with open(ann_file, "r") as file_reader:
        lines = file_reader.readlines()

    data_list = []
    for i, line in enumerate(lines):
        try:
            filename, label = line.split(",")
            label = label.strip("\n")
            with open(os.path.join(img_dir, filename), "rb") as file_reader:
                img = file_reader.read()
            data_json = {"id": i,
                         "file_name": filename,
                         "data": img,
                         "label": int(label)}
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


@pytest.fixture(name="remove_mindrecord_file")
def add_and_remove_cv_file():
    """add/remove cv file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
        yield "yield_cv_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


def test_Mindrecord_Padded(remove_mindrecord_file):
    """
    Feature: PaddedDataset
    Description: Test padding an MindDataset with PaddedDataset
    Expectation: Output is equal to the expected output
    """
    result_list = []
    verify_list = [[1, 2], [3, 4], [5, 11], [6, 12], [7, 13], [8, 14], [9], [10]]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", ['file_name'], num_readers, shuffle=False)
    data1 = [{'file_name': np.array('image_00011.jpg', dtype=np.str_)},
             {'file_name': np.array('image_00012.jpg', dtype=np.str_)},
             {'file_name': np.array('image_00013.jpg', dtype=np.str_)},
             {'file_name': np.array('image_00014.jpg', dtype=np.str_)}]
    ds1 = ds.PaddedDataset(data1)
    ds2 = data_set + ds1
    shard_num = 8
    for i in range(shard_num):
        testsampler = ds.DistributedSampler(num_shards=shard_num, shard_id=i, shuffle=False, num_samples=None)
        ds2.use_sampler(testsampler)
        tem_list = []
        for ele in ds2.create_dict_iterator(num_epochs=1, output_numpy=True):
            tem_list.append(int(ele['file_name'].item().lstrip('image_').rstrip('.jpg')))
        result_list.append(tem_list)
    assert result_list == verify_list


def test_clue_padded_and_skip_with_0_samples():
    """
    Feature: PaddedDataset
    Description: Test padding a CLUEDataset with PaddedDataset with and without samples
    Expectation: Output is equal to the expected output except when dataset has no samples, in which error is raised
    """
    TRAIN_FILE = '../data/dataset/testCLUE/afqmc/train.json'

    data = ds.CLUEDataset(TRAIN_FILE, task='AFQMC', usage='train')
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 3

    data_copy1 = copy.deepcopy(data)

    sample = {"label": np.array(1, np.str_),
              "sentence1": np.array(1, np.str_),
              "sentence2": np.array(1, np.str_)}
    samples = [sample]
    padded_ds = ds.PaddedDataset(samples)
    dataset = data + padded_ds
    testsampler = ds.DistributedSampler(num_shards=2, shard_id=1, shuffle=False, num_samples=None)
    dataset.use_sampler(testsampler)
    assert dataset.get_dataset_size() == 2
    count = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2

    dataset = dataset.skip(count=2)  # dataset2 has none samples
    count = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 0

    with pytest.raises(ValueError, match="There are no samples in the "):
        dataset = dataset.concat(data_copy1)
        count = 0
        for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == 2


def test_celeba_padded():
    """
    Feature: PaddedDataset
    Description: Test padding an CelebADataset with PaddedDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.CelebADataset("../data/dataset/testCelebAData/")

    padded_samples = [{'image': np.zeros(1, np.uint8), 'attr': np.zeros(1, np.uint32)}]
    padded_ds = ds.PaddedDataset(padded_samples)
    data = data + padded_ds
    dis_sampler = ds.DistributedSampler(num_shards=2, shard_id=1, shuffle=False, num_samples=None)
    data.use_sampler(dis_sampler)
    data = data.repeat(2)

    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count = count + 1
    assert count == 4


def test_imagefolder_padded_debug_mode():
    """
    Feature: PaddedDataset
    Description: When dataset is empty, iterate over the data when num_epochs is not 1 in debug mode
    Expectation: Output is equal to the expected output
    """
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)
    data_dir = "../data/dataset/testPK/data"
    data = ds.ImageFolderDataset(data_dir)

    data1 = [{'image': np.zeros(1, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(2, np.uint8), 'label': np.array(1, np.int32)},
             {'image': np.zeros(3, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(4, np.uint8), 'label': np.array(1, np.int32)},
             {'image': np.zeros(5, np.uint8), 'label': np.array(0, np.int32)},
             {'image': np.zeros(6, np.uint8), 'label': np.array(1, np.int32)}]

    data2 = ds.PaddedDataset(data1)
    data3 = data + data2
    testsampler = ds.DistributedSampler(num_shards=51, shard_id=50, shuffle=False, num_samples=None)
    data3.use_sampler(testsampler)
    assert sum([1 for _ in data3]) == 0
    verify_list = []

    for ele in data3.create_dict_iterator(num_epochs=2, output_numpy=True):
        verify_list.append(len(ele['image']))
    assert not verify_list
    ds.config.set_debug_mode(debug_mode_original)


if __name__ == '__main__':
    test_TFRecord_Padded()
    test_GeneratorDataSet_Padded()
    test_Reapeat_afterPadded()
    test_bath_afterPadded()
    test_Unevenly_distributed()
    test_three_datasets_connected()
    test_raise_error()
    test_imagefolder_padded()
    test_more_shard_padded()
    test_Mindrecord_Padded(add_and_remove_cv_file)
    test_imagefolder_padded_debug_mode()
