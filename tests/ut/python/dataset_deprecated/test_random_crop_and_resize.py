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
Testing RandomCropAndResize op in DE
"""
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset as ds
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger


DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_random_crop_and_resize_callable_numpy():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize C++ op is callable with NumPy input
    Expectation: Passes the shape equality test
    """
    logger.info("test_random_crop_and_resize_callable_numpy")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = c_vision.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    # test one tensor
    random_crop_and_resize_op1 = c_vision.RandomResizedCrop(size=(256, 512), scale=(2, 2), ratio=(1, 3),
                                                            interpolation=Inter.AREA)
    img1 = random_crop_and_resize_op1(img)
    assert img1.shape == (256, 512, 3)


def test_random_crop_and_resize_callable_pil():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize Python op is callable with PIL input
    Expectation: Passes the shape equality test
    """
    logger.info("test_random_crop_and_resize_callable_pil")

    img = Image.open("../data/dataset/apple.jpg").convert("RGB")

    assert img.size == (4032, 2268)

    # test one tensor
    random_crop_and_resize_op1 = py_vision.RandomResizedCrop(size=(256, 512), scale=(2, 2), ratio=(1, 3),
                                                             interpolation=Inter.ANTIALIAS)
    img1 = random_crop_and_resize_op1(img)
    assert img1.size == (512, 256)


def test_random_crop_and_resize_op_py_antialias():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize with Python transformations where image interpolation mode
        is Inter.ANTIALIAS
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_crop_and_resize_op_py_antialias")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # With these inputs we expect the code to crop the whole image
    transforms1 = [
        py_vision.Decode(),
        py_vision.RandomResizedCrop((256, 512), (2, 2), (1, 3), Inter.ANTIALIAS),
        py_vision.ToTensor()
    ]
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 3
    logger.info("use RandomResizedCrop by Inter.ANTIALIAS process {} images.".format(num_iter))


def test_random_crop_and_resize_eager_error_02():
    """
    Feature: RandomCropAndResize op
    Description: Test RandomCropAndResize Python op in eager mode with NumPy input and
        Python only interpolation ANTIALIAS
    Expectation: Correct error is thrown as expected
    """
    img = np.random.randint(0, 1, (100, 100, 3)).astype(np.uint8)
    with pytest.raises(TypeError) as error_info:
        random_crop_and_resize_op = py_vision.RandomResizedCrop(size=(100, 200), scale=[1.0, 2.0],
                                                                interpolation=Inter.ANTIALIAS)
        _ = random_crop_and_resize_op(img)
    assert "img should be PIL image. Got <class 'numpy.ndarray'>." in str(error_info.value)


if __name__ == "__main__":
    test_random_crop_and_resize_callable_numpy()
    test_random_crop_and_resize_callable_pil()
    test_random_crop_and_resize_op_py_antialias()
    test_random_crop_and_resize_eager_error_02()
