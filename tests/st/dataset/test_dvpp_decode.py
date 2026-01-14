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
Testing DVPP Decode operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.transforms as t_trans
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    data_dir6 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    data_list.append(data_dir1)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    data_list.append(data_dir6)
    return data_list


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
def test_dvpp_decode_operation_01():
    """
    Feature: Decode operation on device
    Description: Testing the normal functionality of the Decode operator on device
    Expectation: The Output is equal to the expected output
    """
    # decode operator: Test input is jpg, as type is uint8
    image = np.fromfile(dir_data()[1], dtype=np.uint8)
    decode_op = v_trans.Decode().device(device_target="Ascend")
    decode_op_cpu = v_trans.Decode()(image)
    out = decode_op(image)
    assert (out == decode_op_cpu).all()

    # decode operator: Test parameters to_pil is False
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    decode_op = v_trans.Decode().device(device_target="Ascend")
    dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
    for _ in dataset2.create_dict_iterator():
        pass

    # decode operator: The input data has been decoded.
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    with pytest.raises(RuntimeError,
                       match=r"Invalid data shape. Currently only support 1D. Its rank is: 3"):
        decode_op = v_trans.Decode().device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
        for _ in dataset2.create_dict_iterator():
            pass

    # decode operator: Test normal.
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(to_pil=False).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(to_pil=False),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds2 = ds2.map(input_columns=["image"], operations=transform)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True),
                            ds2.create_dict_iterator(output_numpy=True)):
        image1 = data1["image"]
        image2 = data2["image"]
        assert (image2 == image1).all()

    # decode operator: to_pil is True,input is np(jpg)
    image = np.fromfile(dir_data()[1], dtype=np.uint8)
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    decode_op_cpu = v_trans.Decode()(image)
    out = decode_op(image)
    assert (out == decode_op_cpu).all()

    # decode operator: dvpp operator does not support (png)
    image = np.fromfile(dir_data()[3], dtype=np.uint8)
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Invalid image type. Currently only support JPG."):
        _ = decode_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_decode_exception_01():
    """
    Feature: Decode operation on device
    Description: Testing the Decode Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # decode operator: Test error decode
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    with pytest.raises(RuntimeError,
                       match=" Invalid data shape. Currently only support 1D. Its rank is: 3"):  # Error support dimension is different
        decode_op = v_trans.Decode().device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
        for _ in dataset2.create_dict_iterator():
            pass

    # decode operator: Test input is jpg, as type is int32
    image = np.fromfile(dir_data()[1], dtype=np.int32)  # Only supports int8
    decode_op = v_trans.Decode().device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Invalid data type. Currently only support uint8. Its type is: int32"):
        decode_op(image)

    # decode operator: Test input is list
    image = np.fromfile(dir_data()[1], dtype=np.float16).tolist()
    decode_op = v_trans.Decode().device(device_target="Ascend")
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'list'>."):
        decode_op(image)

    # decode operator: Test input is Tensor
    image = Tensor(np.fromfile(dir_data()[1], dtype=np.float16))
    decode_op = v_trans.Decode().device(device_target="Ascend")
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'mindspore.common.tensor.Tensor'>."):
        decode_op(image)

    # decode operator: Test input is 2d
    image = np.random.randn(64, 3)
    decode_op = v_trans.Decode().device(device_target="Ascend")
    with pytest.raises(TypeError, match="The number of array dimensions of the encoded image should be 1, but got 2."):
        decode_op(image)

    # decode operator: No input data passed
    decode_op = v_trans.Decode().device(device_target="Ascend")
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        decode_op()

    # decode operator: Input data passed as 1
    decode_op = v_trans.Decode().device(device_target="Ascend")
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'int'>."):
        decode_op(1)

    # decode operator: to_pil is True,input is np
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    with pytest.raises(RuntimeError,
                       match="Invalid data shape. Currently only support 1D. Its rank is: 3"):
        transforms = [
            v_trans.Decode(to_pil=False).device(device_target="Ascend"),
            v_trans.ToTensor()
        ]
        transform = t_trans.Compose(transforms)
        ds1 = ds1.map(input_columns=["image"], operations=transform)

        for _ in ds1.create_dict_iterator(output_numpy=True):
            pass

    # decode operator: Test One more parameter
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(TypeError, match="too many positional arguments"):
        transforms = [
            v_trans.Decode(True, True).device(device_target="Ascend"),
            v_trans.ToTensor()
        ]
        transform = t_trans.Compose(transforms)
        ds1 = ds1.map(input_columns=["image"], operations=transform)

        for _ in ds1.create_dict_iterator(output_numpy=True):
            pass

    # decode operator: Test input is numpy
    image = np.random.randint(0, 255, (128,)).astype(np.uint8)
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Invalid image type. Currently only support JPG."):
        _ = decode_op(image)

    # decode operator: Test input is PIL image
    with Image.open(dir_data()[1]) as image:
        decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
        with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                            "but got <class 'PIL.JpegImagePlugin.JpegImageFile'>."):
            decode_op(image)

    # decode operator: Test don't have input
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        decode_op()

    # decode operator: Test input is int
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'int'>."):
        decode_op(10)

    # decode operator: Test input is list
    image = np.fromfile(dir_data()[1], dtype=np.uint8).tolist()
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>, "
                                        "but got <class 'list'>."):
        decode_op(image)

    # decode operator: Test parameters to_pil is True
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError,
                       match=r'The transform "Decode\(to_pil=True\)" cannot be performed on Ascend device'):
        decode_op = v_trans.Decode(to_pil=True).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=decode_op)
        for _ in dataset2.create_dict_iterator():
            pass

    # decode operator: Test parameters to_pil is 1
    with pytest.raises(TypeError,
                       match=r"Argument to_pil with value 1 is not of "
                             r"type \[<class 'bool'>\], but got <class 'int'>."):
        v_trans.Decode(to_pil=1).device(device_target="Ascend")

    # decode operator: Test parameters to_pil is 'a'
    with pytest.raises(TypeError,
                       match=r"Argument to_pil with value a is not of "
                             r"type \[<class 'bool'>\], but got <class 'str'>."):
        v_trans.Decode(to_pil='a').device(device_target="Ascend")

    # decode operator: Test parameters to_pil is None
    with pytest.raises(TypeError,
                       match=r"Argument to_pil with value None is not of "
                             r"type \[<class 'bool'>\], but got <class 'NoneType'>."):
        v_trans.Decode(to_pil=None).device(device_target="Ascend")

    # decode operator: to_pil is True,input is np(bmg)
    image = np.fromfile(dir_data()[2], dtype=np.uint8)
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Invalid image type. Currently only support JPG."):
        _ = decode_op(image)

    # decode operator: to_pil is True,input is np(gif)
    image = np.fromfile(dir_data()[4], dtype=np.uint8)
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Invalid image type. Currently only support JPG."):
        _ = decode_op(image)

    # decode operator: Test to_pil is 1.1
    with pytest.raises(TypeError, match=r"Argument to_pil with value 1.1 is not of "
                                        r"type \[<class 'bool'>\], but got <class 'float'>"):
        v_trans.Decode(to_pil=1.1).device(device_target="Ascend")


if __name__ == '__main__':
    test_dvpp_decode_operation_01()
    test_dvpp_decode_exception_01()
