# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import numpy as np
import matplotlib.pyplot as plt

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

FLICKR30K_DATASET_DIR = "../data/dataset/testFlickrData/flickr30k/flickr30k-images"
FLICKR30K_ANNOTATION_FILE_1 = "../data/dataset/testFlickrData/flickr30k/test1.token"
FLICKR30K_ANNOTATION_FILE_2 = "../data/dataset/testFlickrData/flickr30k/test2.token"


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    plt.figure(figsize=(10, 10))
    for i, item in enumerate(zip(images, labels), start=1):
        plt.imshow(item[0])
        plt.title('\n'.join([s.decode('utf-8') for s in item[1]]))
        plt.savefig('./flickr_' + str(i) + '.jpg')


def test_flickr30k_dataset_train(plot=False):
    """
    Feature: FlickrDataset
    Description: Test train for FlickrDataset
    Expectation: The dataset is processed as expected
    """
    data = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_1, decode=True)
    count = 0
    images_list = []
    annotation_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("item[image] is {}".format(item["image"]))
        images_list.append(item['image'])
        annotation_list.append(item['annotation'])
        count = count + 1
    assert count == 2
    if plot:
        visualize_dataset(images_list, annotation_list)


def test_flickr30k_dataset_annotation_check():
    """
    Feature: FlickrDataset
    Description: Test annotation for FlickrDataset
    Expectation: The dataset is processed as expected
    """
    data = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_1, decode=True, shuffle=False)
    count = 0
    expect_annotation_arr = [
        np.array([
            r'This is \*a banana.',
            'This is a yellow banana.',
            'This is a banana on the table.',
            'The banana is yellow.',
            'The banana is very big.',
        ]),
        np.array([
            'This is a pen.',
            'This is a red and black pen.',
            'This is a pen on the table.',
            'The color of the pen is red and black.',
            'The pen has two colors.',
        ])
    ]
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(item["annotation"], expect_annotation_arr[count])
        logger.info("item[image] is {}".format(item["image"]))
        count = count + 1
    assert count == 2


def test_flickr30k_dataset_basic():
    """
    Feature: FlickrDataset
    Description: Test basic parameters and methods of FlickrDataset
    Expectation: The dataset is processed as expected
    """
    # case 1: test num_samples
    data1 = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_2, num_samples=2, decode=True)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter1 += 1
    assert num_iter1 == 2

    # case 2: test repeat
    data2 = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_1, decode=True)
    data2 = data2.repeat(5)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter2 += 1
    assert num_iter2 == 10

    # case 3: test batch with drop_remainder=False
    data3 = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_2, decode=True, shuffle=False)
    resize_op = vision.Resize((100, 100))
    data3 = data3.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    assert data3.get_dataset_size() == 3
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=2)  # drop_remainder is default to be False
    assert data3.get_dataset_size() == 2
    assert data3.get_batch_size() == 2
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1
    assert num_iter3 == 2

    # case 4: test batch with drop_remainder=True
    data4 = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_2, decode=True, shuffle=False)
    resize_op = vision.Resize((100, 100))
    data4 = data4.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    assert data4.get_dataset_size() == 3
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=2, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data4.get_dataset_size() == 1
    assert data4.get_batch_size() == 2
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter4 += 1
    assert num_iter4 == 1


def test_flickr30k_dataset_exception():
    """
    Feature: FlickrDataset
    Description: Test invalid parameters for FlickrDataset
    Expectation: Correct error is thrown as expected
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_1, decode=True)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.FlickrDataset(FLICKR30K_DATASET_DIR, FLICKR30K_ANNOTATION_FILE_1, decode=True)
        data = data.map(operations=exception_func, input_columns=["annotation"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_flickr30k_dataset_train(False)
    test_flickr30k_dataset_annotation_check()
    test_flickr30k_dataset_basic()
    test_flickr30k_dataset_exception()
