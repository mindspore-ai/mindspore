# Copyright 2020 Huawei Technologies Co., Ltd
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
Testing RandomChoice op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger
from util import visualize, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_choice_op(plot=False):
    """
    Test RandomChoice in python transformations
    """
    logger.info("test_random_choice_op")
    # define map operations
    transforms_list = [py_vision.CenterCrop(64), py_vision.RandomRotation(30)]
    transforms1 = [
        py_vision.Decode(),
        py_vision.RandomChoice(transforms_list),
        py_vision.ToTensor()
    ]
    transform1 = py_vision.ComposeOp(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = py_vision.ComposeOp(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(input_columns=["image"], operations=transform1())
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform2())

    image_choice = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_choice.append(image1)
        image_original.append(image2)
    if plot:
        visualize(image_original, image_choice)


def test_random_choice_comp(plot=False):
    """
    Test RandomChoice and compare with single CenterCrop results
    """
    logger.info("test_random_choice_comp")
    # define map operations
    transforms_list = [py_vision.CenterCrop(64)]
    transforms1 = [
        py_vision.Decode(),
        py_vision.RandomChoice(transforms_list),
        py_vision.ToTensor()
    ]
    transform1 = py_vision.ComposeOp(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.CenterCrop(64),
        py_vision.ToTensor()
    ]
    transform2 = py_vision.ComposeOp(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(input_columns=["image"], operations=transform1())
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform2())

    image_choice = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_choice.append(image1)
        image_original.append(image2)

        mse = diff_mse(image1, image2)
        assert mse == 0
    if plot:
        visualize(image_original, image_choice)


def test_random_choice_exception_random_crop_badinput():
    """
    Test RandomChoice: hit error in RandomCrop with greater crop size,
    expected to raise error
    """
    logger.info("test_random_choice_exception_random_crop_badinput")
    # define map operations
    # note: crop size[5000, 5000] > image size[4032, 2268]
    transforms_list = [py_vision.RandomCrop(5000)]
    transforms = [
        py_vision.Decode(),
        py_vision.RandomChoice(transforms_list),
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)
    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(input_columns=["image"], operations=transform())
    try:
        _ = data.create_dict_iterator().get_next()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Crop size" in str(e)


if __name__ == '__main__':
    test_random_choice_op(plot=True)
    test_random_choice_comp(plot=True)
    test_random_choice_exception_random_crop_badinput()
