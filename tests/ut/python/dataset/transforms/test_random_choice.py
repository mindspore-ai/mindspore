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
"""
Test RandomChoice op in Dataset
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
from mindspore.dataset import vision
from mindspore import log as logger
from util import visualize_list, diff_mse, config_get_set_seed

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_choice_c():
    """
    Feature: RandomChoice Op
    Description: Test C++ implementation, both valid and invalid input
    Expectation: Dataset pipeline runs successfully and results are verified for valid input.
        Invalid input is detected.
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=data_trans.RandomChoice(op_list), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # Test whether an operation would be randomly chosen.
    # In order to prevent random failure, both results need to be checked.
    res1 = test_config([[0, 1, 2]], [data_trans.PadEnd([4], 0), data_trans.Slice([0, 2])])
    assert res1 in [[[0, 1, 2, 0]], [[0, 2]]]

    # Test nested structure
    res2 = test_config([[0, 1, 2]], [data_trans.Compose([data_trans.Duplicate(), data_trans.Concatenate()]),
                                     data_trans.Compose([data_trans.Slice([0, 1]), data_trans.OneHot(2)])])
    assert res2 in [[[[1, 0], [0, 1]]], [[0, 1, 2, 0, 1, 2]]]
    # Test RandomChoice when there is only 1 operation
    assert test_config([[4, 3], [2, 1]], [data_trans.Slice([0])]) == [[4], [2]]

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_random_choice_op(plot=False):
    """
    Feature: RandomChoice op
    Description: Test RandomChoice op in Python implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_choice_op")
    # define map operations
    transforms_list = [vision.CenterCrop(64), vision.RandomRotation(30)]
    transforms1 = [
        vision.Decode(True),
        data_trans.RandomChoice(transforms_list),
        vision.ToTensor()
    ]
    transform1 = data_trans.Compose(transforms1)

    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = data_trans.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_choice = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_choice.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_choice)


def test_random_choice_comp(plot=False):
    """
    Feature: RandomChoice op
    Description: Test RandomChoice op and compare with single CenterCrop results
    Expectation: Resulting datasets are expected to be equal
    """
    logger.info("test_random_choice_comp")
    # define map operations
    transforms_list = [vision.CenterCrop(64)]
    transforms1 = [
        vision.Decode(True),
        data_trans.RandomChoice(transforms_list),
        vision.ToTensor()
    ]
    transform1 = data_trans.Compose(transforms1)

    transforms2 = [
        vision.Decode(True),
        vision.CenterCrop(64),
        vision.ToTensor()
    ]
    transform2 = data_trans.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_choice = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_choice.append(image1)
        image_original.append(image2)

        mse = diff_mse(image1, image2)
        assert mse == 0
    if plot:
        visualize_list(image_original, image_choice)


def test_random_choice_exception_random_crop_badinput():
    """
    Feature: RandomChoice op
    Description: Test RandomChoice op where error in RandomCrop occurs due to greater crop size
    Expectation: Error is raised as expected
    """
    logger.info("test_random_choice_exception_random_crop_badinput")
    # define map operations
    # note: crop size[5000, 5000] > image size[4032, 2268]
    transforms_list = [vision.RandomCrop(5000)]
    transforms = [
        vision.Decode(True),
        data_trans.RandomChoice(transforms_list),
        vision.ToTensor()
    ]
    transform = data_trans.Compose(transforms)
    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])
    try:
        _ = data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Crop size" in str(e)


if __name__ == '__main__':
    test_random_choice_c()
    test_random_choice_op(plot=True)
    test_random_choice_comp(plot=True)
    test_random_choice_exception_random_crop_badinput()
