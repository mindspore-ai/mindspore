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
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_solarize_op(threshold=None, plot=False):
    """
    Test RandomSolarize
    """
    logger.info("Test RandomSolarize")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    decode_op = vision.Decode()

    if threshold is None:
        solarize_op = vision.RandomSolarize()
    else:
        solarize_op = vision.RandomSolarize(threshold)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=solarize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    image_solarized = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image_solarized.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize_list(image, image_solarized)


def test_random_solarize_md5():
    """
    Test RandomSolarize
    """
    logger.info("Test RandomSolarize")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_solarize_op = vision.RandomSolarize((10, 150))
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_solarize_op)
    # Compare with expected md5 from images
    filename = "random_solarize_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_solarize_errors():
    """
    Test that RandomSolarize errors with bad input
    """
    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((12, 1))
    assert "threshold must be in min max format numbers" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        vision.RandomSolarize((12, 1000))
    assert "Input is not within the required interval of (0 to 255)." in str(error_info.value)

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
    test_random_solarize_op((100, 100), plot=True)
    test_random_solarize_op((12, 120), plot=True)
    test_random_solarize_op(plot=True)
    test_random_solarize_errors()
    test_random_solarize_md5()
