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
Test RandomApply op in Dataset
"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
from mindspore.dataset import vision
from mindspore import log as logger
from util import visualize_list, config_get_set_seed, \
    config_get_set_num_parallel_workers, save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_apply_c():
    """
    Feature: RandomApply Op
    Description: Test C++ implementation, both valid and invalid input
    Expectation: Dataset pipeline runs successfully and results are verified for valid input.
        Invalid input is detected.
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, op_list, prob=0.5):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=data_trans.RandomApply(op_list, prob), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    res1 = test_config([[0, 1]], [data_trans.Duplicate(), data_trans.Concatenate()])
    assert res1 in [[[0, 1]], [[0, 1, 0, 1]]]
    # test single nested compose
    assert test_config([[0, 1, 2]], [
        data_trans.Compose([data_trans.Duplicate(), data_trans.Concatenate(), data_trans.Slice([0, 1, 2])])]) == \
           [[0, 1, 2]]
    assert test_config([[0, 1, 2]], [
        data_trans.Compose(
            [data_trans.Duplicate(), data_trans.Concatenate(), lambda x: x, data_trans.Slice([0, 1, 2])])]) == \
           [[0, 1, 2]]
    # test exception
    assert "is not of type [<class 'list'>]" in test_config([1, 0], data_trans.TypeCast(mstype.int32))
    assert "Input prob is not within the required interval" in test_config([0, 1], [data_trans.Slice([0, 1])], 1.1)
    assert "is not of type [<class 'float'>, <class 'int'>]" in test_config([1, 0], [data_trans.TypeCast(mstype.int32)],
                                                                            None)
    assert "transforms list with value None is not of type [<class 'list'>]" in test_config([1, 0], None)
    assert "is neither a transforms op (TensorOperation) nor a callable pyfunc" in \
           test_config([[0, 1, 2]], [data_trans.Duplicate(), data_trans.Concatenate(), "zyx"])

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_random_apply_op(plot=False):
    """
    Feature: RandomApply op
    Description: Test RandomApply in Python transformations
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_apply_op")
    # define map operations
    transforms_list = [vision.CenterCrop(64), vision.RandomRotation(30)]
    transforms1 = [
        vision.Decode(True),
        data_trans.RandomApply(transforms_list, prob=0.6),
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

    image_apply = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_apply.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_apply)


def test_random_apply_md5():
    """
    Feature: RandomApply op
    Description: Test RandomApply op with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("test_random_apply_md5")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms_list = [vision.CenterCrop(64), vision.RandomRotation(30)]
    transforms = [
        vision.Decode(True),
        # Note: using default value "prob=0.5"
        data_trans.RandomApply(transforms_list),
        vision.ToTensor()
    ]
    transform = data_trans.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_apply_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_apply_exception_random_crop_badinput():
    """
    Feature: RandomApply op
    Description: Test RandomApply with invalid input for one of the transform functions
    Expectation: Correct error is raised as expected
    """
    logger.info("test_random_apply_exception_random_crop_badinput")
    original_seed = config_get_set_seed(200)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms_list = [vision.Resize([32, 32]),
                       vision.RandomCrop(100),  # crop size > image size
                       vision.RandomRotation(30)]
    transforms = [
        vision.Decode(True),
        data_trans.RandomApply(transforms_list, prob=0.6),
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
    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


if __name__ == '__main__':
    test_random_apply_c()
    test_random_apply_op(plot=True)
    test_random_apply_md5()
    test_random_apply_exception_random_crop_badinput()
