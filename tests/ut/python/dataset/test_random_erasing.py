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
Testing RandomErasing op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_image, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_random_erasing_op(plot=False):
    """
    Test RandomErasing and Cutout
    """
    logger.info("test_random_erasing")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = mindspore.dataset.transforms.py_transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.Cutout(80)
    ]
    transform_2 = mindspore.dataset.transforms.py_transforms.Compose(transforms_2)
    data2 = data2.map(operations=transform_2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        mse = diff_mse(image_1, image_2)
        if plot:
            visualize_image(image_1, image_2, mse)


def test_random_erasing_md5():
    """
    Test RandomErasing with md5 check
    """
    logger.info("Test RandomErasing with md5 check")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = mindspore.dataset.transforms.py_transforms.Compose(transforms_1)
    data = data.map(operations=transform_1, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_erasing_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


if __name__ == "__main__":
    test_random_erasing_op(plot=True)
    test_random_erasing_md5()
