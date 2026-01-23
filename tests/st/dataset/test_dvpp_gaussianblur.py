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
Testing DVPP GaussianBlur operation
"""
import os
import pytest
import numpy as np
import cv2
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.transforms.transforms as t_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=102.0)
def test_dvpp_gaussianblur_operation_01():
    """
    Feature: GaussianBlur operation on device
    Description: Testing the normal functionality of the GaussianBlur operator on device
    Expectation: The Output is equal to the expected output
    """
    # GaussianBlur operator: Test sigma is 102.6
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    gaussianblur_op = v_trans.GaussianBlur([5, 5], 102.6).device(device_target="Ascend")
    gaussianblur_op_cpu = v_trans.GaussianBlur([5, 5], 102.6)
    dataset1 = dataset1.map(input_columns=["image"], operations=gaussianblur_op)
    dataset2 = dataset2.map(input_columns=["image"], operations=gaussianblur_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1)

    # Using GaussianBlur operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.GaussianBlur([5, 5], 102.6).device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.GaussianBlur([5, 5], 102.6)(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], atol=1, rtol=1)

    # GaussianBlur operator: Test sigma is 2
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    gaussianblur_op = v_trans.GaussianBlur(1, 2).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur(1, 2)(image)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    gaussianblur_op = v_trans.GaussianBlur((5, 5), 2).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur((5, 5), 2)(image)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    gaussianblur_op = v_trans.GaussianBlur((1, 1), 2).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur((1, 1), 2)(image)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    gaussianblur_op = v_trans.GaussianBlur((5, 1), 2).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur((5, 1), 2)(image)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    # GaussianBlur operator: Test gaussian_kernel is (3,3)
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    gaussianblur_op = v_trans.GaussianBlur((3, 3), 2).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur((3, 3), 2)(image)
    assert np.allclose(gaussianblur_op, gaussianblur_op_cpu, rtol=1)

    gaussianblur_op = v_trans.GaussianBlur((1, 3), 2).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur((1, 3), 2)(image)
    assert np.allclose(gaussianblur_op, gaussianblur_op_cpu, rtol=1)

    # GaussianBlur operator: Test sigma is 120
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    gaussianblur_op = v_trans.GaussianBlur((5, 5), 120).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur((5, 5), 120)(image)
    assert np.allclose(gaussianblur_op, gaussianblur_op_cpu, rtol=1)

    # GaussianBlur operator: Test sigma is [1.2, 3.5]
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    img_array = np.array(image)
    gaussianblur_op = v_trans.GaussianBlur((1, 1), [1.2, 3.5]).device(device_target="Ascend")(img_array)
    gaussianblur_op_cpu = v_trans.GaussianBlur((1, 1), [1.2, 3.5])(img_array)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()
    image.close()

    # GaussianBlur operator: Test sigma is [16.0, 0.1]
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_bmp)
    gaussianblur_op = v_trans.GaussianBlur([5, 5], [16.0, 0.1]).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur([5, 5], [16.0, 0.1])(image)
    assert np.allclose(gaussianblur_op, gaussianblur_op_cpu, rtol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=36.0)
def test_dvpp_gaussianblur_operation_02():
    """
    Feature: GaussianBlur operation on device
    Description: Testing the normal functionality of the GaussianBlur operator on device
    Expectation: The Output is equal to the expected output
    """
    # GaussianBlur operator: Test sigma is 0
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    img_array = np.array(image)
    gaussianblur_op = v_trans.GaussianBlur(1, 0).device(device_target="Ascend")(img_array)
    gaussianblur_op_cpu = v_trans.GaussianBlur(1, 0)(img_array)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()
    image.close()

    # GaussianBlur operator: Test sigma is none
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    gaussianblur_op = v_trans.GaussianBlur(5).device(device_target="Ascend")
    out = gaussianblur_op(image)
    gaussianblur_op = v_trans.GaussianBlur(5, 1.1).device(device_target="Ascend")
    out2 = gaussianblur_op(image)
    assert (out == out2).all()

    # GaussianBlur operator: Test input.shape is (50, 50, 3)
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    gaussianblur_op = v_trans.GaussianBlur(1, 100.0).device(device_target="Ascend")(image)
    gaussianblur_op_cpu = v_trans.GaussianBlur(1, 100.0)(image)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    # GaussianBlur operator: Test input.shape is (50, 50, 1)
    image = np.random.randint(0, 255, (50, 50, 1)).astype(np.uint8)
    gaussianblur_op = v_trans.GaussianBlur(1, 100.0).device(device_target="Ascend")(image)
    new_arr = np.reshape(image, (50, 50))
    gaussianblur_op_cpu = v_trans.GaussianBlur(1, 100.0)(new_arr)
    assert gaussianblur_op.shape == (50, 50)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    image = np.random.randint(0, 255, (1, 50, 50, 3)).astype(np.uint8)
    gaussianblur_op = v_trans.GaussianBlur(1, 100.0).device(device_target="Ascend")(image)
    assert gaussianblur_op.shape == (50, 50, 3)

    # GaussianBlur operator: Test input is 4d
    image = np.random.randint(0, 255, (1, 50, 50, 3)).astype(np.uint8)
    gaussianblur_op = v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")(image)
    new_arr = np.reshape(image, (50, 50, 3))
    gaussianblur_op_cpu = v_trans.GaussianBlur(1, 0.8)(new_arr)
    assert (gaussianblur_op == gaussianblur_op_cpu).all()

    # GaussianBlur operator: Test input is 3d list
    image = list(np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        gaussianblur_op = v_trans.GaussianBlur(1, 0).device(device_target="Ascend")
        _ = gaussianblur_op(image)

    # GaussianBlur operator: Test input is 3d tuple
    image = tuple(np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        gaussianblur_op = v_trans.GaussianBlur(1, 0).device(device_target="Ascend")
        _ = gaussianblur_op(image)

    # GaussianBlur operator: more operations
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.GaussianBlur(1, 0)
    ]
    ds1 = ds1.map(input_columns=["image"], operations=transforms)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        v_trans.Decode().device(device_target="Ascend"),
        v_trans.GaussianBlur(1, 0).device(device_target="Ascend")
    ]
    ds2 = ds2.map(input_columns=["image"], operations=transforms1)
    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        assert (data1["image"] == data2["image"]).all()

    # GaussianBlur operator: Test GaussianBlur data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "")
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.GaussianBlur(1, 0),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        v_trans.Decode().device(device_target="Ascend"),
        v_trans.GaussianBlur(1, 0).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        assert (data1["image"] == data2["image"]).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_gaussianblur_exception_01():
    """
    Feature: GaussianBlur operation on device
    Description: Testing the GaussianBlur Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # GaussianBlur operator: kernel_size is 2
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    with pytest.raises(RuntimeError, match="When target is Ascend, `kernel_size` only supports values 1, 3, and 5."):
        gaussianblur_op = v_trans.GaussianBlur(11, 2).device(device_target="Ascend")
        gaussianblur_op(image)

    # GaussianBlur operator: Test input.shape is (50, 50, 8)
    image = np.random.randint(0, 255, (50, 50, 8)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=" The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3"):
        _ = v_trans.GaussianBlur(1, 100.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test input.shape is (50, 50, 2)
    image = np.random.randint(0, 255, (50, 50, 2)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=" The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3"):
        _ = v_trans.GaussianBlur(1, 100.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test input.shape is (50, 50, 3, 3)
    image = np.random.randint(0, 255, (50, 50, 3, 3)).astype(np.uint8)
    gaussianblur_op = v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match=r"The input tensor NHWC should be 1HWC or HWC."):
        gaussianblur_op(image)

    # GaussianBlur operator: Test input is numpy array of the int64 type
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.int64)
    gaussianblur_op = v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")
    with pytest.raises(RuntimeError) as error_log:
        gaussianblur_op(image)
        assert "The input data is not uint8 or float32" in str(error_log.value)

    # GaussianBlur operator: Test input is 1d
    image = np.random.randint(0, 255, (50,)).astype(np.uint8)
    gaussianblur_op = v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match=r"DvppGaussianBlur: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        gaussianblur_op(image)

    # GaussianBlur operator: Test input image is (1, 50, 3, 3)
    image = np.random.randint(0, 255, (1, 50, 3, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[8192, 4096\\], but got \\[50, 3\\]"):
        v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")(image)

    image = np.random.randint(0, 255, (1, 3, 50, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[8192, 4096\\], but got \\[3, 50\\]"):
        v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")(image)

    image = np.random.randint(0, 255, (1, 50, 4097, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[8192, 4096\\], but got \\[50, 4097\\]"):
        v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test input is int
    image = 10
    gaussianblur_op = v_trans.GaussianBlur(1, 0.8).device(device_target="Ascend")
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'int'>"):
        gaussianblur_op(image)

    # GaussianBlur operator: Test no Parameters
    with pytest.raises(TypeError, match="missing a required argument: 'kernel_size'"):
        v_trans.GaussianBlur().device(device_target="Ascend")

    # GaussianBlur operator: Test more Parameters
    with pytest.raises(TypeError, match="too many positional arguments"):
        v_trans.GaussianBlur(1, 1.0, 3).device(device_target="Ascend")

    # GaussianBlur operator: Test pepiline kernel_size is 2
    with pytest.raises(ValueError, match="Input kernel_size is not an odd value."):
        v_trans.GaussianBlur(2, 1.0).device(device_target="Ascend")

    # GaussianBlur operator: Test kernel_size is 1-list
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Kernel size should be a single integer or a "
                                        "list/tuple \\(kernel_width, kernel_height\\) of length 2."):
        v_trans.GaussianBlur([1], 1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test kernel_size is -1
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(ValueError, match="Input kernel_size is not within the required interval of \\[1, 16777216\\]."):
        v_trans.GaussianBlur(-1, 1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test kernel_size is 3-tuple
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Kernel size should be a single integer or a "
                                        "list/tuple \\(kernel_width, kernel_height\\) of length 2."):
        v_trans.GaussianBlur((1, 3, 5), 1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test kernel_size is np
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument kernel_size with value \\[1 3\\] is not of type \\[\\<class "
                                        "'int'\\>, \\<class 'list'\\>, \\<class 'tuple'\\>\\], but got \\<class "
                                        "'numpy.ndarray'\\>."):
        v_trans.GaussianBlur(np.array([1, 3]), 1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test kernel_size is float
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument kernel_size\\[1\\] with value 3.0 is not "
                                        "of type \\[\\<class 'int'\\>\\], but got \\<class 'float'\\>."):
        v_trans.GaussianBlur([3, 3.0], 1.0).device(device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_gaussianblur_exception_02():
    """
    Feature: GaussianBlur operation on device
    Description: Testing the GaussianBlur Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # GaussianBlur operator: Test kernel_size > 16777216
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(ValueError, match="Input kernel_size is not within the required interval of \\[1, 16777216\\]."):
        v_trans.GaussianBlur(16777217, 1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test kernel_size is str
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument kernel_size with value 3 is not of type \\[\\<class 'int'\\>, "
                                        "\\<class 'list'\\>, \\<class 'tuple'\\>\\], but got \\<class 'str'\\>."):
        v_trans.GaussianBlur("3", 1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test sigma < 0
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(ValueError, match="Input sigma is not within the required interval of \\[0, 16777216\\]."):
        v_trans.GaussianBlur(3, -1.0).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test sigma > 16777216
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(ValueError, match="Input sigma is not within the required interval of \\[0, 16777216\\]."):
        v_trans.GaussianBlur(3, 16777216.1).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test sigma is str
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument sigma with value 3 is not of type \\[\\<class 'numbers.Number'\\>, "
                                        "\\<class 'list'\\>, \\<class 'tuple'\\>\\], but got \\<class 'str'\\>."):
        v_trans.GaussianBlur(3, "3").device(device_target="Ascend")(image)

    # GaussianBlur operator: Test sigma is 1-list
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Sigma should be a single number or a list/tuple of length 2 for width and height."):
        v_trans.GaussianBlur(3, [0.5]).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test sigma is 3-tuple
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Sigma should be a single number or a list/tuple of length 2 for width and height."):
        v_trans.GaussianBlur(3, (0.1, 0.8, 0.6)).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test sigma is np
    image = np.random.randint(0, 255, (50, 50, 3)).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument sigma with value \\[0.1 0.8\\] is not of type \\[\\<class "
                                        "'numbers.Number'\\>, \\<class 'list'\\>, \\<class 'tuple'\\>\\], but "
                                        "got \\<class 'numpy.ndarray'\\>."):
        v_trans.GaussianBlur(3, np.array([0.1, 0.8])).device(device_target="Ascend")(image)

    # GaussianBlur operator: Test no input
    gaussianblur_op = v_trans.GaussianBlur(1, 0).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        gaussianblur_op()

    # When the input image is Pillow, the Affine interface call fails
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend, "
                                            "you can convert the input to the numpy ndarray type"):
            _ = v_trans.GaussianBlur(1, 100.0).device(device_target="Ascend")(image)


if __name__ == '__main__':
    test_dvpp_gaussianblur_operation_01()
    test_dvpp_gaussianblur_operation_02()
    test_dvpp_gaussianblur_exception_01()
    test_dvpp_gaussianblur_exception_02()
