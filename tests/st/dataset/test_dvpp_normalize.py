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
Testing DVPP Normalize operation
"""
import os
import numpy as np
import pytest
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=42.0)
def test_dvpp_normalize_operation_01():
    """
    Feature: Normalize operation on device
    Description: Testing the normal functionality of the Normalize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Normalize operator, normal test, Normalize operator heterogeneous and non-heterogeneous results are consistent
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ms.set_seed(1)
    meanr = 121.0
    meang = 115.0
    meanb = 100.0
    stdr = 70
    stdg = 68
    stdb = 71
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset1 = dataset1.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"),
                            offload=False)
    dataset2 = dataset2.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"),
                            offload=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert image.shape == image_aug.shape
        assert np.allclose(image, image_aug)

    # Normalize operator, normal test, mean is list(int)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    meanr = 121
    meang = 115
    meanb = 100
    stdr = 70
    stdg = 68
    stdb = 71
    mean = [meanr, meang, meanb]
    std = (stdr, stdg, stdb)
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset1 = dataset1.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std))
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = dataset2.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"))
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert image.shape == image_aug.shape
        assert np.allclose(image, image_aug)

    # Normalize operator, normal test, std is tuple(int)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    meanr = 121
    meang = 115
    meanb = 100
    stdr = 70
    stdg = 68
    stdb = 70
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset1 = dataset1.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std))
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = dataset2.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"))
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert image.shape == image_aug.shape
        assert np.allclose(image, image_aug)

    # Normalize operator, normal test, std is tuple(float)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    meanr = 121.0
    meang = 115.0
    meanb = 100.0
    stdr = 70.0
    stdg = 68.0
    stdb = 71.0
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset1 = dataset1.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std))
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = dataset2.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"))
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert image.shape == image_aug.shape
        assert np.allclose(image, image_aug)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=44.0)
def test_dvpp_normalize_operation_02():
    """
    Feature: Normalize operation on device
    Description: Testing the normal functionality of the Normalize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Normalize operator, normal test, mean and std are tuple(float), small values
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    meanr = 1.5
    meang = 1.0
    meanb = 0.3
    stdr = 1.5
    stdg = 1.0
    stdb = 0.5
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset1 = dataset1.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std))
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = dataset2.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"))
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert image.shape == image_aug.shape
        assert np.allclose(image, image_aug)

    # Normalize operator, normal test, std is list(int)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    meanr = 121
    meang = 115
    meanb = 100
    stdr = 70.0
    stdg = 68
    stdb = 71
    mean = [meanr, meang, meanb]
    std = [stdr, stdg, stdb]
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset1 = dataset1.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std))
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = dataset2.map(input_columns=["image"],
                            operations=vision.Normalize(mean=mean, std=std).device(device_target="Ascend"))
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert image.shape == image_aug.shape
        assert np.allclose(image, image_aug)

    # Normalize operator, normal test, eager mode, input data is numpy <H,W,1>
    mean = (255.0,)
    std = (0.1,)
    image = np.random.randn(100, 100, 1).astype("uint8")  # ascend only supports float32,uint8
    normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
    out = normalize_op(image)
    assert out.shape == (100, 100)  # ascend:(100, 100)

    image = np.random.randn(100, 100, 3).astype("uint8")
    normalize_op = vision.Normalize(mean=[121, 115, 100], std=[70.0, 68, 71]).device(device_target="Ascend")
    normalize_op_cpu = vision.Normalize(mean=[121, 115, 100], std=[70.0, 68, 71])(image)
    out = normalize_op(image)
    assert np.allclose(out, normalize_op_cpu)

    # Normalize operator, normal test, eager mode, input data is numpy <H,W,3>
    mean = (255.0, 0.0, 255.0)
    std = (0.1, 1.2, 0.2)
    image = np.random.randn(100, 100, 3).astype("float32")
    normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
    normalize_op_cpu = vision.Normalize(mean=mean, std=std)(image)
    out = normalize_op(image)
    assert np.allclose(out, normalize_op_cpu, rtol=0.001, atol=0.001)

    # Normalize operator, normal test, eager mode, input data is numpy <1,H,W>
    mean = (255.0,)
    std = (0.1,)
    image = np.random.randn(1, 100, 100).astype("float32")
    normalize_op = vision.Normalize(mean=mean, std=std, is_hwc=False).device(device_target="Ascend")
    out = normalize_op(image)  # shape missing 1
    assert out.shape == (100, 100)

    # Normalize operator, normal test, eager mode, input data is numpy <3,H,W>
    mean = (255.0, 0.0, 255.0)
    std = (0.1, 1.2, 0.2)
    image = np.random.randn(3, 100, 100).astype("float32")
    normalize_op = vision.Normalize(mean=mean, std=std, is_hwc=False).device(device_target="Ascend")
    normalize_op_cpu = vision.Normalize(mean=mean, std=std, is_hwc=False)(image)
    out = normalize_op(image)
    assert out.shape == (3, 100, 100)
    assert np.allclose(out, normalize_op_cpu)

    # Normalize operator, normal test, mean and std length is 1, fill all channels
    mean = [50]
    std = [32.0]
    image = np.random.randn(64, 64, 3).astype("float32")
    with pytest.raises(RuntimeError,
                       match="DvppNormalize: The channel is not equal to the size of mean or std."):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")  # ascend does not support
        normalize_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_normalize_exception_01():
    """
    Feature: Normalize operation on device
    Description: Testing the Normalize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Normalize operator, exception test, element value in mean is -1
    meanr = -1
    meang = 115
    meanb = 100
    stdr = 70
    stdg = 68
    stdb = 71
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    with pytest.raises(ValueError,
                       match=r"Input mean\[0]\ is not within the required interval of \[0, 255\]."):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, element type error in mean
    meanr = ""
    meang = 115
    meanb = 100
    stdr = 70
    stdg = 68
    stdb = 71
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    with pytest.raises(TypeError, match=r"Argument mean\[0\] with value \"\" is not of "
                                        r"type \[<class 'int'>, <class 'float'>\], but got <class 'str'>."):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, length of mean and std are inconsistent
    stdr = 70
    stdg = 68
    stdb = 71
    mean = (100, 100)
    std = (stdr, stdg, stdb)
    with pytest.raises(ValueError,
                       match="Length of mean and std must be equal"):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, mean type error str
    stdr = 70
    stdg = 68
    stdb = 71
    mean = ""
    std = (stdr, stdg, stdb)
    with pytest.raises(TypeError,
                       match='Argument mean with value "" is not of type'):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, mean type error int
    stdr = 70
    stdg = 68
    stdb = 71
    mean = 1
    std = (stdr, stdg, stdb)
    with pytest.raises(TypeError,
                       match="Argument mean with value 1 is not of type"):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, element value in std is -1
    meanr = 121
    meang = 115
    meanb = 100
    stdr = -1
    stdg = 68
    stdb = 71
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    with pytest.raises(ValueError,
                       match=r"Input std\[0\] is not within the required interval of \(0, 255\]"):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, element type error in std
    meanr = 120
    meang = 115
    meanb = 100
    stdr = ""
    stdg = 68
    stdb = 71
    mean = (meanr, meang, meanb)
    std = (stdr, stdg, stdb)
    with pytest.raises(TypeError, match=r"Argument std\[0\] with value \"\" is not of "
                                        r"type \[<class 'int'>, <class 'float'>\], but got <class 'str'>."):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, std type error str
    mean = (100, 100, 100)
    std = ""
    with pytest.raises(TypeError,
                       match='Argument std with value "" is not of type'):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, std type error int
    mean = (100, 100, 100)
    std = 1
    with pytest.raises(TypeError,
                       match="Argument std with value 1 is not of type"):
        vision.Normalize(mean=mean, std=std).device(device_target="Ascend")

    # Normalize operator, exception test, no parameters passed
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.Normalize().device(device_target="Ascend")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_normalize_exception_02():
    """
    Feature: Normalize operation on device
    Description: Testing the Normalize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Normalize operator, exception test, missing mean parameter
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.Normalize(std=(10, 10, 10)).device(device_target="Ascend")

    # Normalize operator, exception test, missing std parameter
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.Normalize(mean=(10, 10, 10)).device(device_target="Ascend")

    # Normalize operator, normal test, eager mode, input data is numpy <H,W,2>
    mean = (255.0, 0.0)
    std = (0.1, 1.2)
    image = np.random.randn(100, 100, 2).astype("float32")  # channel number 2 not supported
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[H,W,C\] is not 1, 3, but got: 2"):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)

    # Normalize operator, normal test, eager mode, input data is numpy <H,W,4>
    mean = (255.0, 0.0, 255.0, 0.0)
    std = (0.1, 1.2, 0.2, 1.2)
    image = np.random.randn(100, 100, 4).astype("float32")
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[H,W,C\] is not 1, 3, but got: 4"):
        normalize_op = vision.Normalize(mean=mean, std=std, is_hwc=True).device(device_target="Ascend")
        normalize_op(image)

    # Normalize operator, normal test, eager mode, input data is numpy <2,H,W>
    mean = (255.0, 0.0)
    std = (0.1, 1.2)
    image = np.random.randn(2, 100, 100).astype("float32")
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[C,H,W\] is not 1 or 3, but got: 2"):
        normalize_op = vision.Normalize(mean=mean, std=std, is_hwc=False).device(device_target="Ascend")
        normalize_op(image)  # channel number 2 not supported

    # Normalize operator, exception test, eager mode, input data is numpy <4,H,W>
    mean = (255.0, 0.0, 255.0, 0.0)
    std = (0.1, 1.2, 0.2, 1.2)
    image = np.random.randn(4, 100, 100).astype("float32")
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[C,H,W\] is not 1 or 3, but got: 4"):
        normalize_op = vision.Normalize(mean=mean, std=std, is_hwc=False).device(device_target="Ascend")
        normalize_op(image)  # channel number 4 not supported

    # Normalize operator, exception test, element value in mean is 255.1
    mean = (255.1, 150.0, 126.4)
    std = (0.6, 85.0, 122.0)
    image = np.random.randn(64, 128, 3)
    with pytest.raises(ValueError,
                       match=r"Input mean\[0\] is not within the required interval of \[0, 255\]."):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)

    mean = (255.0,)
    std = (0.1,)
    image = np.random.randn(8193, 100, 1).astype("uint8")
    normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[8192, 4096\\]"):
        _ = normalize_op(image)

    # Normalize operator, exception test, element value in std is 0.0
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_2.jpg")
    mean = (56.0, 150.0, 126.4)
    std = (0.0, 85.0, 122.0)
    with Image.open(image_file) as image:
        with pytest.raises(ValueError,
                           match=r"Input std\[0\] is not within the required interval of \(0, 255\]."):
            normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
            normalize_op(image)

    # Normalize operator, exception test, mean, std and input data channel number do not match
    mean = (56.0, 150.0)
    std = (85.0, 122.0)
    image = np.random.randn(64, 128, 3)
    with pytest.raises(RuntimeError,
                       match="DvppNormalize: The channel is not equal to the size of mean or std."):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)

    # Normalize operator, exception test, input data is numpy tolist()
    mean = [56.0, 150.0, 60.0]
    std = [85.0, 122.0, 80.0]
    image = np.random.randn(64, 128, 3).tolist()
    normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        normalize_op(image)

    # Normalize operator, exception test, mean is ms.Tensor
    mean = Tensor([56.0, 150.0, 100.0])
    std = [85.0, 122.0, 80.0]
    image = np.random.randn(64, 128, 3).astype("float32")
    with pytest.raises(TypeError, match="is not of type "):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_normalize_exception_03():
    """
    Feature: Normalize operation on device
    Description: Testing the Normalize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Normalize operator, exception test, mean and std length are inconsistent
    mean = [56.0, 150.0, 60.0]
    std = [0.1, 0.8]
    image = np.random.randn(64, 128, 3)
    with pytest.raises(ValueError,
                       match="Length of mean and std must be equal."):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)

    # Normalize operator, exception test, input data is 4-dimensional
    mean = [56.0, 150.0, 60.0]
    std = [0.1, 0.8, 1.2]
    image = np.random.randn(32, 32, 3, 3)  # CPU supports up to 4-dimensional input, ascend does not support and reports error
    with pytest.raises(RuntimeError,
                       match="The input tensor NHWC should be 1HWC or HWC."):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)

    # Normalize operator, exception test, input data is missing
    mean = [56.0, 150.0, 60.0]
    std = [0.1, 0.8, 1.2]
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op()

    # Normalize operator, exception test, input data is 5-dimensional
    mean = [56.0, 150.0, 60.0]
    std = [0.1, 0.8, 1.2]
    image = np.random.randn(32, 32, 32, 3, 3).astype("float32")  # ascend does not support 5-dimensional
    with pytest.raises(RuntimeError,
                       match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]"):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")  # ascend does not support
        normalize_op(image)

    # Normalize operator, exception test, input type unsupported type
    mean = (255.0,)
    std = (0.1,)
    image = np.random.randn(100, 100, 1).astype("uint16")  # ascend only supports float32,uint8
    with pytest.raises(RuntimeError,
                       match=r"DvppNormalize: The input data is not uint8 or float32."):
        normalize_op = vision.Normalize(mean=mean, std=std).device(device_target="Ascend")
        normalize_op(image)


if __name__ == '__main__':
    test_dvpp_normalize_operation_01()
    test_dvpp_normalize_operation_02()
    test_dvpp_normalize_exception_01()
    test_dvpp_normalize_exception_02()
    test_dvpp_normalize_exception_03()
