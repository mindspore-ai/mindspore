# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing FiveCrop in DE
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(image_1, image_2):
    """
    visualizes the image using FiveCrop
    """
    plt.subplot(161)
    plt.imshow(image_1)
    plt.title("Original")

    for i, image in enumerate(image_2):
        image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
        plt.subplot(162 + i)
        plt.imshow(image)
        plt.title("image {} in FiveCrop".format(i + 1))

    plt.show()


def skip_test_five_crop_op():
    """
    Test FiveCrop
    """
    logger.info("test_five_crop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(),
        vision.ToTensor(),
    ]
    transform_1 = vision.ComposeOp(transforms_1)
    data1 = data1.map(input_columns=["image"], operations=transform_1())

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(),
        vision.FiveCrop(200),
        lambda images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 5 images
    ]
    transform_2 = vision.ComposeOp(transforms_2)
    data2 = data2.map(input_columns=["image"], operations=transform_2())

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = item2["image"]

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        # visualize(image_1, image_2)

        # The output data should be of a 4D tensor shape, a stack of 5 images.
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5


def test_five_crop_error_msg():
    """
    Test FiveCrop error message.
    """
    logger.info("test_five_crop_error_msg")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(),
        vision.FiveCrop(200),
        vision.ToTensor()
    ]
    transform = vision.ComposeOp(transforms)
    data = data.map(input_columns=["image"], operations=transform())

    with pytest.raises(RuntimeError) as info:
        data.create_tuple_iterator().get_next()
    error_msg = "TypeError: img should be PIL Image or Numpy array. Got <class 'tuple'>"

    # error msg comes from ToTensor()
    assert error_msg in str(info.value)


if __name__ == "__main__":
    test_five_crop_op()
    test_five_crop_error_msg()
