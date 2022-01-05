# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


DATA_DIR = "../data/dataset/testPK/data"


def test_offload():
    """
    Feature: test map offload flag.
    Description: Input is image dataset.
    Expectation: Output should be same with activated or deactivated offload.
    """
    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(8, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(8, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(img_0, img_1)


def test_auto_offload():
    """
    Feature: Test auto_offload config option.
    Description: Input is image dataset.
    Expectation: Output should same with auto_offload activated and deactivated.
    """
    trans = [C.Decode(), C.HWC2CHW()]

    # Enable automatic offload
    ds.config.set_auto_offload(True)

    # Dataset with offload deactivated
    dataset_auto_disabled = ds.ImageFolderDataset(DATA_DIR)
    dataset_auto_disabled = dataset_auto_disabled.map(operations=trans, input_columns="image", offload=False)
    dataset_auto_disabled = dataset_auto_disabled.batch(8, drop_remainder=True)

    # Dataset with config.auto_offload activated
    dataset_auto_enabled = ds.ImageFolderDataset(DATA_DIR)
    dataset_auto_enabled = dataset_auto_enabled.map(operations=trans, input_columns="image")
    dataset_auto_enabled = dataset_auto_enabled.batch(8, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_auto_disabled.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_auto_enabled.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(img_0, img_1)

    # Need to turn off here or subsequent test cases will fail.
    ds.config.set_auto_offload(False)


def test_offload_concat_dataset_1():
    """
    Feature: test map offload flag for concatenated dataset.
    Description: Input is image dataset.
    Expectation: Should raise RuntimeError.
    """
    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(8, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(8, drop_remainder=True)

    dataset_concat = dataset_0 + dataset_1

    error_msg = "Offload module currently does not support concatenated or zipped datasets."
    with pytest.raises(RuntimeError, match=error_msg):
        for (_, _) in dataset_concat.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue


def test_offload_concat_dataset_2():
    """
    Feature: test map offload flag for concatenated dataset.
    Description: Input is image dataset.
    Expectation: Should raise RuntimeError.
    """
    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")

    dataset_concat = dataset_0 + dataset_1
    dataset_concat = dataset_concat.batch(8, drop_remainder=True)

    error_msg = "Offload module currently does not support concatenated or zipped datasets."
    with pytest.raises(RuntimeError, match=error_msg):
        for (_, _) in dataset_concat.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue


def test_offload_normalize_op():
    """
    Feature: test map offload Normalize op.
    Description: Input is image dataset.
    Expectation: Output should be same with activated or deactivated offload for Normalize op.
    """
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.Normalize(mean=mean, std=std)], input_columns="image", offload=True)
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(8, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.Normalize(mean=mean, std=std)], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(8, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)


def test_offload_rescale_op():
    """
    Feature: test map offload Rescale op.
    Description: Input is image dataset.
    Expectation: Output should be same with activated or deactivated offload for Rescale op.
    """
    rescale = 1.0 / 255.0
    shift = 0.0

    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.Rescale(rescale, shift)], input_columns="image", offload=True)
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(8, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.Rescale(rescale, shift)], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(8, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)


def test_offload_different_column_end_of_pipeline():
    """
    Feature: Test offload end_of_pipeline check.
    Description: Input is image dataset.
    Expectation: The image map op gets offloaded even though it comes before the not-offloaded label map op, since
                 the end_of_pipeline check looks at columns separately.
    """
    image_trans = [C.Decode(), C.HWC2CHW()]
    ds.config.set_auto_offload(True)

    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=image_trans, input_columns="image")
    dataset_0 = dataset_0.map(operations=[C2.TypeCast(mstype.int32)], input_columns="label", offload=False)

    data_iterator = dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True)
    # Assert at least one operation has been offloaded
    np.testing.assert_(len(data_iterator.offload_model.transform_list[0].me_ops) > 0)

    ds.config.set_auto_offload(False)


def test_offload_not_end_of_pipeline():
    """
    Feature: Test offload end_of_pipeline check.
    Description: Input is image dataset.
    Expectation: No operations are offloaded, since the image map op at the end of the pipeline has the
                 offload flag set to False.
    """
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image", offload=True)
    dataset_0 = dataset_0.map(operations=[C.RandomHorizontalFlip(prob=0.5)], input_columns="image", offload=True)
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=False)

    dataset_0 = dataset_0.map(operations=[C2.TypeCast(mstype.int32)], input_columns="label", offload=False)

    data_iterator = dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True)
    # Assert no operations are set to be offloaded
    np.testing.assert_(data_iterator.offload_model is None)


def test_offload_dim_check():
    """
    Feature: test input has the required number of dimensions for offload operation.
    Description: Input is image dataset.
    Expectation: Should raise ValueError.
    """
    # Dataset with offload activated.
    dataset = ds.ImageFolderDataset(DATA_DIR)
    dataset = dataset.map(operations=[C.Decode()], input_columns="image")
    dataset = dataset.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)

    error_msg = "For HwcToChw offload operation, the dimension of input should be 4, but got 3."
    with pytest.raises(ValueError, match=error_msg):
        for (_, _) in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue


if __name__ == "__main__":
    test_offload()
    test_auto_offload()
    test_offload_concat_dataset_1()
    test_offload_concat_dataset_2()
    test_offload_normalize_op()
    test_offload_rescale_op()
    test_offload_different_column_end_of_pipeline()
    test_offload_not_end_of_pipeline()
    test_offload_dim_check()
