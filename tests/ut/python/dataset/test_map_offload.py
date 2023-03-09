# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

DATA_DIR = "../data/dataset/testPK/data"
BATCH_SIZE = 2


def test_offload():
    """
    Feature: Test map offload flag.
    Description: Input is image dataset.
    Expectation: Output should be same with activated or deactivated offload.
    """
    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(BATCH_SIZE, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(BATCH_SIZE, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(img_0, img_1)
        break


def test_offload_string():
    """
    Feature: Test map offload flag with string tensors.
    Description: Input is text dataset.
    Expectation: Output should be same with activated or deactivated offload (incl. decoded text).
    """

    # Dataset with offload activated.
    data0 = ds.TextFileDataset("../data/dataset/testVocab/words.txt", shuffle=False)

    # Dataset with offload not activated.
    data1 = ds.TextFileDataset("../data/dataset/testVocab/words.txt", shuffle=False)

    # Use Data Transforms PadEnd op in operations list for Map
    padend_op = C2.PadEnd([100], pad_value='<pad>')

    data0 = data0.map(operations=[padend_op], input_columns=["text"], offload=True)
    data1 = data1.map(operations=[padend_op], input_columns=["text"])

    for d0, d1 in zip(data0.create_dict_iterator(num_epochs=1, output_numpy=True),
                      data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d0['text'], (d1['text']))


@pytest.mark.forked
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
    dataset_auto_disabled = dataset_auto_disabled.batch(BATCH_SIZE, drop_remainder=True)

    # Dataset with config.auto_offload activated
    dataset_auto_enabled = ds.ImageFolderDataset(DATA_DIR)
    dataset_auto_enabled = dataset_auto_enabled.map(operations=trans, input_columns="image")
    dataset_auto_enabled = dataset_auto_enabled.batch(BATCH_SIZE, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_auto_disabled.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_auto_enabled.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(img_0, img_1)
        break

    # Need to turn off here or subsequent test cases will fail.
    ds.config.set_auto_offload(False)


def test_offload_column_validation():
    """
    Feature: Test the column validation for offloaded map operations
    Description: Input is an image dataset, but the input column is incorrect for the offloaded map operation.
    Expectation: Should raise RuntimeError.
    """
    dataset = ds.ImageFolderDataset(DATA_DIR)
    dataset = dataset.map(operations=[C.Decode()], input_columns="image")
    # Use invalid input column name
    dataset = dataset.map(operations=[C.HWC2CHW()], input_columns="fake_column", offload=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    error_msg = "The following input column(s) for an offloaded map operation do not exist: [\'fake_column\']"
    with pytest.raises(RuntimeError) as excinfo:
        for (_, _) in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue
    assert str(excinfo.value) == error_msg


def test_offload_multi_column():
    """
    Feature: Test the offload functionality with datasets with more than 2 columns.
    Description: Input is an image dataset, copy the image column and apply map operations to both images.
    Expectation: Output should be same with both offload activated and deactivated.
    """

    def copy_column(x, y):
        return x, x, y

    dataset = ds.ImageFolderDataset(DATA_DIR)
    dataset = dataset.map(operations=copy_column, input_columns=["image", "label"],
                          output_columns=["image1", "image2", "label"])
    dataset = dataset.map(operations=[C.Decode()], input_columns="image1")
    dataset = dataset.map(operations=[C.HWC2CHW()], input_columns="image1")
    dataset = dataset.map(operations=[C.Decode()], input_columns="image2")
    dataset = dataset.map(operations=[C.HWC2CHW()], input_columns="image2")
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset_offload = ds.ImageFolderDataset(DATA_DIR)
    dataset_offload = dataset_offload.map(operations=copy_column, input_columns=["image", "label"],
                                          output_columns=["image1", "image2", "label"])
    dataset_offload = dataset_offload.map(operations=[C.Decode()], input_columns="image1")
    dataset_offload = dataset_offload.map(operations=[C.HWC2CHW()], input_columns="image1", offload=True)
    dataset_offload = dataset_offload.map(operations=[C.Decode()], input_columns="image2")
    dataset_offload = dataset_offload.map(operations=[C.HWC2CHW()], input_columns="image2", offload=True)
    dataset_offload = dataset_offload.batch(BATCH_SIZE, drop_remainder=True)

    for (img1, img2, _), (img1_offload, img2_offload, _) in \
        zip(dataset.create_tuple_iterator(num_epochs=1, output_numpy=True),
                dataset_offload.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(img1, img1_offload)
        np.testing.assert_array_equal(img2, img2_offload)
        break


def test_offload_column_mapping():
    """
    Feature: Test the dataset column mapping for offloaded operations
    Description: Input is an image dataset, copy the image column, then apply offload to only copied column.
    Expectation: The offload model dataset column index value is 1 (second column).
    """

    def copy_column(x, y):
        return x, x, y

    dataset = ds.ImageFolderDataset(DATA_DIR)
    dataset = dataset.map(operations=copy_column, input_columns=["image", "label"],
                          output_columns=["image1", "image2", "label"])
    dataset = dataset.map(operations=[C.Decode()], input_columns="image2")
    dataset = dataset.map(operations=[C.HWC2CHW()], input_columns="image2", offload=True)

    dataset_iterator = dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)

    offload_col_idxs = dataset_iterator.offload_model.transform_list[0].col_idxs
    # assert there is only one column index in the offload model, and that it is 1 (second column)
    np.testing.assert_((len(offload_col_idxs) == 1) and (offload_col_idxs[0] == 1))


def test_offload_concat_dataset_1():
    """
    Feature: Test map offload flag for concatenated dataset.
    Description: Input is image dataset.
    Expectation: Should raise RuntimeError.
    """
    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(BATCH_SIZE, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(BATCH_SIZE, drop_remainder=True)

    dataset_concat = dataset_0 + dataset_1

    error_msg = "Offload module currently does not support concatenated or zipped datasets."
    with pytest.raises(RuntimeError, match=error_msg):
        for (_, _) in dataset_concat.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue


def test_offload_concat_dataset_2():
    """
    Feature: Test map offload flag for concatenated dataset.
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
    dataset_concat = dataset_concat.batch(BATCH_SIZE, drop_remainder=True)

    error_msg = "Offload module currently does not support concatenated or zipped datasets."
    with pytest.raises(RuntimeError, match=error_msg):
        for (_, _) in dataset_concat.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue


def test_offload_normalize_op():
    """
    Feature: Test map offload Normalize op.
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
    dataset_0 = dataset_0.batch(BATCH_SIZE, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.Normalize(mean=mean, std=std)], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(BATCH_SIZE, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)
        break


def test_offload_rescale_op():
    """
    Feature: Test map offload Rescale op.
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
    dataset_0 = dataset_0.batch(BATCH_SIZE, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.Rescale(rescale, shift)], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(BATCH_SIZE, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)
        break


def test_offload_typecast_op():
    """
    Feature: Test map offload TypeCast op.
    Description: Input is image dataset.
    Expectation: Output should be the same with activated or deactivated offload for TypeCast op.
    """
    # Dataset without offload activated.
    ds_baseline = ds.ImageFolderDataset(DATA_DIR, num_samples=3)
    ds_baseline = ds_baseline.map(operations=[C.Decode(), C2.TypeCast(mstype.float32)], input_columns="image")
    ds_baseline = ds_baseline.map(operations=[C2.TypeCast("int32")], input_columns="label")

    # Dataset with offload activated.
    ds_offload = ds.ImageFolderDataset(DATA_DIR, num_samples=10)
    ds_offload = ds_offload.map(operations=[C.Decode(), C2.TypeCast(mstype.float32)],
                                input_columns="image", offload=True)
    ds_offload = ds_offload.map(operations=[C2.TypeCast("int32")], input_columns="label", offload=True)

    for (img_0, _), (img_1, _) in zip(ds_baseline.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      ds_offload.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)


def test_offload_typecast_op_2():
    """
    Feature: Test map offload TypeCast op.
    Description: Test TypeCast op with numpy data type input, and alias ToType
    Expectation: Output should be the same with activated or deactivated offload for TypeCast op.
    """
    # Dataset without offload activated.
    ds_baseline = ds.ImageFolderDataset(DATA_DIR, num_samples=2)
    ds_baseline = ds_baseline.map(operations=[C.Decode(), C2.TypeCast(np.float32)], input_columns="image")
    ds_baseline = ds_baseline.map(operations=[C.ToType(mstype.int32)], input_columns="label")

    # Dataset with offload activated.
    ds_offload = ds.ImageFolderDataset(DATA_DIR, num_samples=5)
    ds_offload = ds_offload.map(operations=[C.Decode(), C2.TypeCast(np.float32)],
                                input_columns="image", offload=True)
    ds_offload = ds_offload.map(operations=[C.ToType(mstype.int32)], input_columns="label", offload=True)

    for (img_0, _), (img_1, _) in zip(ds_baseline.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      ds_offload.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)


@pytest.mark.forked
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
    Feature: Test input has the required number of dimensions for offload operation.
    Description: Input is image dataset.
    Expectation: Should raise ValueError.
    """
    # Dataset with offload activated.
    dataset = ds.ImageFolderDataset(DATA_DIR)
    dataset = dataset.map(operations=[C.Decode()], input_columns="image")
    dataset = dataset.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)

    with pytest.raises(ValueError):
        for (_, _) in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            continue


def test_offload_random_sharpness_op():
    """
    Feature: Test map offload RandomSharpness op.
    Description: Input is image dataset.
    Expectation: Output should be same with activated or deactivated offload for RandomSharpness op.
    """

    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.RandomSharpness(degrees=[1.0, 1.0])], input_columns="image", offload=True)
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(BATCH_SIZE, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.RandomSharpness(degrees=[1.0, 1.0])], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(BATCH_SIZE, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_almost_equal(img_0, img_1, decimal=6)
        break


def test_offload_with_dict_itr():
    """
    Feature: Test offload
    Description: Test map offload with pyfuncs and dict iterator
    Expectation: Test passes without hangs
    """
    dataset = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=False, num_samples=3)
    dataset = dataset.map(operations=[lambda x: x], input_columns="image", python_multiprocessing=False,
                          num_parallel_workers=1)

    type_cast_op = C2.TypeCast(mstype.int32)
    dataset = dataset.map(operations=type_cast_op, input_columns="label", offload=True, python_multiprocessing=False,
                          num_parallel_workers=1)
    # the test is passing with no hangs when num_epochs is not set
    num = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num += 1

    assert num == 3


if __name__ == "__main__":
    test_offload()
    test_offload_string()
    test_auto_offload()
    test_offload_column_validation()
    test_offload_column_mapping()
    test_offload_multi_column()
    test_offload_concat_dataset_1()
    test_offload_concat_dataset_2()
    test_offload_normalize_op()
    test_offload_rescale_op()
    test_offload_typecast_op()
    test_offload_typecast_op_2()
    test_offload_different_column_end_of_pipeline()
    test_offload_not_end_of_pipeline()
    test_offload_dim_check()
    test_offload_random_sharpness_op()
    test_offload_with_dict_itr()
