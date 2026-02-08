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
Testing DVPP Solarize operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=70.0)
def test_dvpp_solarize_operation_01():
    """
    Feature: Solarize operation on device
    Description: Testing the normal functionality of the Solarize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Solarize operator, normal testing, pipeline mode, threshold=200, input image is numpy
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=True)
    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=True)
    ds2 = ds2.map(operations=vision.Solarize(200).device(device_target="Ascend"), input_columns=["image"])
    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp = abs((data1["image"] >= 200).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator, normal testing, pipeline mode, threshold=120.0, input image is numpy
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=False)]
    ds1 = ds1.map(operations=op_list, input_columns=["image"])

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=False), vision.Solarize(120.0).device(device_target="Ascend")]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp = abs((data1["image"] >= 120.0).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator, normal testing, pipeline mode, threshold=(100, 200), input image is PIL
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    ds1 = ds1.map(operations=vision.Decode(to_pil=True), input_columns=["image"])

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=True), vision.Solarize((100, 200)).device(device_target="Ascend")]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp_1 = (data1["image"] >= 100).astype(np.int32)
        out_exp_2 = (data1["image"] <= 200).astype(np.int32)
        out_exp = abs((out_exp_1 * out_exp_2).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Using the Solarize operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = vision.Decode().device("Ascend")(img_bytes)
        img_ops = vision.Solarize((100, 200)).device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)
        img_ops = vision.Solarize((100, 200))(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec)(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Solarize operator, normal testing, pipeline mode, threshold=(0, 255)
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    ds1 = ds1.map(operations=vision.Decode(to_pil=True), input_columns=["image"])

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=False)
    op_list = [vision.Decode(to_pil=True), vision.Solarize(threshold=(0, 255)).device(device_target="Ascend")]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        out_exp_1 = (data1["image"] >= 0).astype(np.int32)
        out_exp_2 = (data1["image"] <= 255).astype(np.int32)
        out_exp = abs((out_exp_1 * out_exp_2).astype(np.int32) * 255 - data1["image"])
        assert (out_exp == data2["image"]).all()

    # Solarize operator, normal testing, eager mode, input image is numpy, threshold=0
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint8)
    op = vision.Solarize(0)(image)
    op_dvpp = vision.Solarize(0).device(device_target="Ascend")(image)
    assert op_dvpp.shape == (20, 20, 3)
    assert op_dvpp.dtype == np.uint8
    out_exp = abs((image >= 0).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_solarize_operation_02():
    """
    Feature: Solarize operation on device
    Description: Testing the normal functionality of the Solarize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Solarize operator, normal testing, eager mode, input image is uint8, threshold=(0.1, 0.2)
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize((0.1, 0.2))(image)
    op_dvpp = vision.Solarize((0.1, 0.2)).device(device_target="Ascend")(image)
    assert op_dvpp.shape == (20, 20, 3)
    assert op_dvpp.dtype == np.uint8
    assert np.allclose(op, op_dvpp, rtol=1, atol=1)

    # Solarize operator, normal testing, eager mode, input image is float32, threshold=(0.1, 0.2)
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Solarize((0.1, 0.2)).device(device_target="Ascend")(image)
        assert "The input data is not uint8" in str(e.value)

    # Solarize operator, normal testing, eager mode, input image is uint8, threshold=(1, 2)
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize((1, 2))(image)
    op_dvpp = vision.Solarize((0.1, 0.2)).device(device_target="Ascend")(image)
    assert op_dvpp.shape == (20, 20, 3)
    assert op_dvpp.dtype == np.uint8
    out_exp_1 = image >= 1
    out_exp_2 = image <= 2
    out_exp = abs((out_exp_1 * out_exp_2).astype(np.int32) * 255.0 - image)
    assert (op_dvpp == out_exp).all()
    assert (op == op_dvpp).all()

    # Solarize operator, normal testing, eager mode, input image is uint8, threshold=0.2666
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize(0.2666)(image)
    op_dvpp = vision.Solarize(0.2666).device(device_target="Ascend")(image)
    assert op_dvpp.shape == (20, 20, 3)
    assert op_dvpp.dtype == np.uint8
    out_exp = abs((image >= 0.2666).astype(np.int32) * 255.0 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()

    # Solarize operator, normal testing, eager mode, input image is uint8, threshold=1.00001
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize(1.00001)(image)
    op_dvpp = vision.Solarize(1.00001).device(device_target="Ascend")(image)
    assert (op == op_dvpp).all()

    # Solarize operator, normal testing, eager mode, input image is uint8, threshold=255
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize(255)(image)
    op_dvpp = vision.Solarize(255).device(device_target="Ascend")(image)
    assert (op == op_dvpp).all()

    # Solarize operator, normal testing, eager mode, input image is uint8, threshold=100
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize(100)(image)
    op_dvpp = vision.Solarize(100).device(device_target="Ascend")(image)
    out_exp = abs((image >= 100).astype(np.float32) * 255.0 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()

    # Solarize Operator, tested normally, input image is uint16, threshold=10.2
    image = (np.random.randint(0, 256, (20, 20, 3)) / 255.0).astype(np.uint8)
    op = vision.Solarize(10.2)(image)
    op_dvpp = vision.Solarize(10.2).device(device_target="Ascend")(image)
    assert isinstance(op_dvpp, np.ndarray)
    assert op_dvpp.shape == (20, 20, 3)
    image = np.array(image)
    out_exp = abs((image >= 10.2).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()

    # Solarize Operator, tested normally, input image is bmp
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_bmp)
    op_dvpp = vision.Solarize(2).device(device_target="Ascend")(image)
    op = vision.Solarize(2)(image)
    assert np.array(op_dvpp).shape == np.array(image).shape
    image = np.array(image)
    out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()

    # Solarize Operator, tested normally, input image is jpg
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_path)
    op_dvpp = vision.Solarize(2).device(device_target="Ascend")(image)
    op = vision.Solarize(2)(image)
    assert np.array(op_dvpp).shape == np.array(image).shape
    image = np.array(image)
    out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()

    # Solarize Operator, tested normally, input image is PNG, mode=RGB
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_path)
    op_dvpp = vision.Solarize(5).device(device_target="Ascend")(image)
    op = vision.Solarize(5)(image)
    assert np.array(op_dvpp).shape == np.array(image).shape
    image = np.array(image)
    out_exp = abs((image >= 5).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_solarize_operation_03():
    """
    Feature: Solarize operation on device
    Description: Testing the normal functionality of the Solarize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Solarize Operator, tested normally, input image is gif, mode=P
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image_pi = Image.open(image_path)
    img_array = np.array(image_pi)
    op_dvpp = vision.Solarize(2).device(device_target="Ascend")(img_array)
    op = vision.Solarize(2)(img_array)
    assert np.array(op_dvpp).shape == np.array(image_pi).shape
    image = np.array(image_pi)
    out_exp = abs((image >= 2).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()
    image_pi.close()

    # Solarize Operator, tested normally, input image is numpy <H,W,1>
    image = np.random.randint(0, 256, (20, 20, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (20, 20))
    op = vision.Solarize(2)(new_arr)
    op_dvpp = vision.Solarize(2).device(device_target="Ascend")(image)
    assert np.array(op_dvpp).shape == (20, 20)
    assert (op == op_dvpp).all()

    # Solarize Operator, tested normally, input image is a two-dimensional numpy array
    image = np.random.randint(0, 256, (20, 20)).astype(np.uint8)
    op_dvpp = vision.Solarize(20).device(device_target="Ascend")(image)
    op = vision.Solarize(20)(image)
    assert np.array(op_dvpp).shape == image.shape
    out_exp = abs((image >= 20).astype(np.int32) * 255 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()

    # Solarize Operator, tested normally, threshold is list
    image = np.random.randint(0, 256, (20, 20)).astype(np.uint8)
    op = vision.Solarize(threshold=[1, 30])(image)
    op_dvpp = vision.Solarize(threshold=[1, 30]).device(device_target="Ascend")(image)
    assert np.array(op_dvpp).shape == image.shape
    out_exp_1 = image >= 1
    out_exp_2 = image <= 30
    out_exp = abs((out_exp_1 * out_exp_2).astype(np.float32) * 255.0 - image)
    assert (out_exp == op_dvpp).all()
    assert (op == op_dvpp).all()
    assert (op == op_dvpp).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=98.0)
def test_dvpp_solarize_exception_01():
    """
    Feature: Solarize operation on device
    Description: Testing the Solarize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Solarize Operator, anomaly testing, input image is PIL
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend, "
                                        "you can convert the input to the numpy ndarray type"):
        _ = vision.Solarize(10).device(device_target="Ascend")(image)

    # Solarize operator, anomaly testing, input image is PNG, mode=RGBA
    image = (np.random.randint(0, 256, (20, 20, 3, 4)) / 255.0).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape \\[N,H,W,C\\] is not 1, 3, but got: 4"):
        _ = vision.Solarize(10).device(device_target="Ascend")(image)

    image = (np.random.randint(0, 256, (20, 20, 2)) / 255.0).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="he channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 2"):
        _ = vision.Solarize(133).device(device_target="Ascend")(image)

    # Solarize operator, anomaly testing, input image is a one-dimensional numpy array
    image = np.random.randint(0, 256, (20,)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"invalid input shape, only support NHWC input, got rank: 1"):
        _ = vision.Solarize(2).device(device_target="Ascend")(image)

    # Solarize operator, anomaly testing, threshold is str
    with pytest.raises(TypeError, match=r"Argument threshold with value 1 is not of type \[<class 'float'>, "
                                        r"<class 'int'>, <class 'list'>, <class 'tuple'>\], but got <class 'str'>."):
        vision.Solarize(threshold='1').device(device_target="Ascend")

    # Solarize operator, anomaly testing, error size
    image = np.random.randn(3, 3, 3).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[8192, 4096\\], but got \\[3, 3\\]"):
        vision.Solarize(threshold=10).device(device_target="Ascend")(image)

    image = np.random.randn(8193, 4097, 3).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[8192, 4096\\], but got \\[8193, 4097\\]"):
        vision.Solarize(threshold=10).device(device_target="Ascend")(image)

    # Solarize operator, anomaly testing, threshold is None
    with pytest.raises(TypeError, match=r"Argument threshold with value None is not of type \[<class 'float'>, "
                                        r"<class 'int'>, <class 'list'>, <class 'tuple'>\], but "
                                        r"got <class 'NoneType'>."):
        vision.Solarize(threshold=None).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold is -0.00001
    with pytest.raises(ValueError, match=r"Input threshold\[0\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=-0.00001).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold is 255.00001
    with pytest.raises(ValueError, match=r"Input threshold\[0\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=255.00001).device(device_target="Ascend")

    # Solarize operator, anomaly testing, The threshold tuple element value is -0.00001.
    with pytest.raises(ValueError, match=r"Input threshold\[0\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=(-0.00001, 255)).device(device_target="Ascend")

    # Solarize operator, anomaly testing, The threshold tuple element value is 255.00001.
    with pytest.raises(ValueError, match=r"Input threshold\[1\] is not within the required interval of \[0, 255\]."):
        vision.Solarize(threshold=(2, 255.00001)).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold tuple element type error
    with pytest.raises(TypeError, match=r"Argument threshold\[0\] with value True is not of type \(<class 'float'>, "
                                        r"<class 'int'>\), but got <class 'bool'>."):
        vision.Solarize(threshold=(True, 255)).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold=(0.2, 2, 255)
    with pytest.raises(TypeError, match=r"threshold must be a single number or sequence of two numbers."):
        vision.Solarize(threshold=(0.2, 2, 255)).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold=(255,)
    with pytest.raises(TypeError, match=r"threshold must be a single number or sequence of two numbers."):
        vision.Solarize(threshold=(255,)).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold=()
    with pytest.raises(TypeError, match=r"threshold must be a single number or sequence of two numbers."):
        vision.Solarize(threshold=()).device(device_target="Ascend")

    # Solarize operator, anomaly testing, threshold tuple min>max
    with pytest.raises(ValueError, match=r"threshold must be in order of \(min, max\)."):
        vision.Solarize(threshold=(200, 2)).device(device_target="Ascend")

    # Solarize operator, anomaly testing, input image is int
    image = 100
    op = vision.Solarize(2).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        op(image)

    # Solarize operator, anomaly testing, input image is str
    image = '100'
    op = vision.Solarize(2).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'str'>."):
        op(image)

    # Solarize operator, anomaly testing, input image is float
    image = 100.0
    op = vision.Solarize(2).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'float'>."):
        op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_solarize_exception_02():
    """
    Feature: Solarize operation on device
    Description: Testing the Solarize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Solarize operator, anomaly testing, input image is list
    image = np.random.randint(0, 256, (10, 10, 3)).tolist()
    op = vision.Solarize(2).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(image)

    # Solarize operator, anomaly testing, input image is tuple
    image = (1, 2, 3)
    op = vision.Solarize(2).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        op(image)

    # Solarize operator, anomaly testing, input image is ms.Tensor
    image = ms.Tensor(np.random.randint(0, 256, (10, 10, 3)))
    op = vision.Solarize(2).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, "
                                        "got <class 'mindspore.common.tensor.Tensor'>."):
        op(image)


if __name__ == '__main__':
    test_dvpp_solarize_operation_01()
    test_dvpp_solarize_operation_02()
    test_dvpp_solarize_operation_03()
    test_dvpp_solarize_exception_01()
    test_dvpp_solarize_exception_02()
