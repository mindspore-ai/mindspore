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
Testing RandomSolarizeOp op in DE
"""
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, config_get_set_seed, config_get_set_num_parallel_workers, \
    visualize_one_channel_dataset

GENERATE_GOLDEN = False

MNIST_DATA_DIR = "../data/dataset/testMnistData"
DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_solarize_op(threshold=(10, 150), plot=False, run_golden=True):
    """
    Test RandomSolarize
    """
    logger.info("Test RandomSolarize")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    if threshold is None:
        solarize_op = vision.RandomSolarize()
    else:
        solarize_op = vision.RandomSolarize(threshold)

    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=solarize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    if run_golden:
        filename = "random_solarize_01_result.npz"
        save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

    image_solarized = []
    image = []

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_solarized.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize_list(image, image_solarized)

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_solarize_mnist(plot=False, run_golden=True):
    """
    Test RandomSolarize op with MNIST dataset (Grayscale images)
    """

    mnist_1 = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_2 = ds.MnistDataset(dataset_dir=MNIST_DATA_DIR, num_samples=2, shuffle=False)
    mnist_2 = mnist_2.map(operations=vision.RandomSolarize((0, 255)), input_columns="image")

    images = []
    images_trans = []
    labels = []

    for _, (data_orig, data_trans) in enumerate(zip(mnist_1, mnist_2)):
        image_orig, label_orig = data_orig
        image_trans, _ = data_trans
        images.append(image_orig.asnumpy())
        labels.append(label_orig.asnumpy())
        images_trans.append(image_trans.asnumpy())

    if plot:
        visualize_one_channel_dataset(images, images_trans, labels)

    if run_golden:
        filename = "random_solarize_02_result.npz"
        save_and_check_md5(mnist_2, filename, generate_golden=GENERATE_GOLDEN)


def test_random_solarize_errors():
    """
    Test that RandomSolarize errors with bad input
    """
    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((12, 1))
    assert "threshold must be in min max format numbers" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((12, 1000))
    assert "Input is not within the required interval of [0, 255]." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.RandomSolarize((122.1, 140))
    assert "Argument threshold[0] with value 122.1 is not of type (<class 'int'>,)." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((122, 100, 30))
    assert "threshold must be a sequence of two numbers" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((120,))
    assert "threshold must be a sequence of two numbers" in str(error_info.value)


if __name__ == "__main__":
    test_random_solarize_op((10, 150), plot=True, run_golden=True)
    test_random_solarize_op((12, 120), plot=True, run_golden=False)
    test_random_solarize_op(plot=True, run_golden=False)
    test_random_solarize_mnist(plot=True, run_golden=True)
    test_random_solarize_errors()
