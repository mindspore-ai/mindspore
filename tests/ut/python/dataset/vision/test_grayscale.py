# Copyright 2025 Huawei Technologies Co., Ltd
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
Testing Grayscale op
"""
import numpy as np
import os
import pytest
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.transforms.transforms as t_trans


TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_grayscale_operation_01():
    """
    Feature: Grayscale operation
    Description: Testing the normal functionality of the Grayscale operator
    Expectation: The Output is equal to the expected output
    """
    # Grayscale operator: Test grayscale normal
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train", "")
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [
        v_trans.Decode(to_pil=True),
        v_trans.Grayscale(),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Grayscale operator: Test image is jpg
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        grayscale_op = v_trans.Grayscale()
        _ = grayscale_op(image)

    # Grayscale operator: Test image is bmp
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        grayscale_op = v_trans.Grayscale(3)
        _ = grayscale_op(image)

    # Grayscale operator: Test image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        grayscale_op = v_trans.Grayscale(1)
        _ = grayscale_op(image)

    # Grayscale operator: Test image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        grayscale_op = v_trans.Grayscale(3)
        _ = grayscale_op(image)


def test_grayscale_exception_01():
    """
    Feature: Grayscale operation
    Description: Testing the normal functionality of the Grayscale operator
    Expectation: The Output is equal to the expected output
    """
    # Grayscale operator: Test image is numpy array
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image_data:
        image = np.array(image_data)
        grayscale_op = v_trans.Grayscale(3)
        with pytest.raises(TypeError, match="img should be PIL image. Got <class 'numpy.ndarray'>."):
            grayscale_op(image)

    # Grayscale operator: Test image is numpy list
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as iamge_data:
        image = np.array(iamge_data).tolist()
        grayscale_op = v_trans.Grayscale(3)
        with pytest.raises(TypeError, match="img should be PIL image. Got <class 'list'>."):
            grayscale_op(image)

    # Grayscale operator: Test no image is transferred
    grayscale_op = v_trans.Grayscale(3)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        grayscale_op()

    # Grayscale operator: Test num_output_channels is 2
    with pytest.raises(ValueError,
                       match="Number of channels of the output grayscale imageshould be either 1 or 3. Got 2."):
        v_trans.Grayscale(2)

    # Grayscale operator: Test num_output_channels is None
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        with pytest.raises(TypeError, match="Argument num_output_channels with value None is not of type " + \
                                            "\\[<class 'int'>\\], but got <class 'NoneType'>"):
            grayscale_op = v_trans.Grayscale(None)
            grayscale_op(image)

    # Grayscale operator: Test num_output_channels is float
    with pytest.raises(TypeError, match="Argument num_output_channels with value 1.5 is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'float'>"):
        v_trans.Grayscale(1.5)

    # Grayscale operator: Test num_output_channels is list
    with pytest.raises(TypeError, match="Argument num_output_channels with value \\[1, 2\\] is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'list'>"):
        v_trans.Grayscale([1, 2])

    # Grayscale operator: Test num_output_channels is str
    with pytest.raises(TypeError, match="Argument num_output_channels with value test is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'str'>"):
        v_trans.Grayscale('test')

    # Grayscale operator: Test input a parameter that does not exist
    with pytest.raises(TypeError,
                       match="got an unexpected keyword argument"):
        v_trans.Grayscale(3, test='test')
