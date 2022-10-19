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
Testing RandomResize op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, diff_mse, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False

def test_random_resize_op(plot=False):
    """
    Feature: RandomResize op
    Description: Test RandomResize op basic usage
    Expectation: The dataset is processed as expected
    """
    logger.info("Test resize")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)

    # apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])

    data2 = data1.map(operations=resize_op, input_columns=["image"])
    image_original = []
    image_resized = []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_1 = item1["image"]
        image_2 = item2["image"]
        image_original.append(image_1)
        image_resized.append(image_2)
        num_iter += 1
    if plot:
        visualize_list(image_original, image_resized)


def test_random_resize_md5():
    """
    Feature: RandomResize op
    Description: Test RandomResize op with md5 check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomResize with md5 check")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=resize_op, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_resize_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

def test_random_resize_op_1():
    """
    Feature: RandomResize op
    Description: Test RandomResize op with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomResize with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=ops.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    resize_op = vision.RandomResize(10)
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=resize_op, input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1


if __name__ == "__main__":
    test_random_resize_op(plot=True)
    test_random_resize_md5()
    test_random_resize_op_1()
