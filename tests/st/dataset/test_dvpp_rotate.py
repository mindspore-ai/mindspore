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
Testing DVPP Rotate operation
"""
import os
import pytest
import cv2
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.utils as mode
from mindspore import log as logger
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


DATA_DIR = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
IMAGE_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "apple.jpg")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=74.0)
def test_dvpp_rotate_operation_01():
    """
    Feature: Rotate operation on device
    Description: Testing the normal functionality of the Rotate operator on device
    Expectation: The Output is equal to the expected output
    """
    # Using the Rotate operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    degrees = 90
    resample = mode.Inter.BILINEAR
    fill_value = 0
    def pyfunc1(img_bytes):
        img_decode = vision.Decode().device("Ascend")(img_bytes)
        img_ops = vision.Rotate(degrees=degrees, resample=resample, expand=True,
                                fill_value=fill_value).device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)
        img_ops = vision.Rotate(degrees=degrees, resample=resample, expand=True,
                                fill_value=fill_value)(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec)(img_ops)
        return img_normalize
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Rotate operator: degrees = 100
    image = cv2.imread(image_png)
    degrees = 100
    resample = mode.Inter.BILINEAR
    fill_value = 255
    img_transformed_ascend = vision.Rotate(degrees=degrees, resample=resample, expand=True,
                                           fill_value=fill_value).device("Ascend")(image)
    image_save = os.path.join(TEST_DATA_DATASET_FUNC, "rotate_process", "dvpp_image.PNG")

    image2 = cv2.imread(image_save, cv2.IMREAD_UNCHANGED)
    assert (img_transformed_ascend == image2).all()

    # Rotate operator: input is png
    image = cv2.imread(image_png)
    degrees = 50
    resample = mode.Inter.BILINEAR
    fill_value = 100
    img_transformed_ascend = vision.Rotate(degrees=degrees, resample=resample, expand=True,
                                           fill_value=fill_value).device("Ascend")(image)
    image_save = os.path.join(TEST_DATA_DATASET_FUNC, "rotate_process", "dvpp_image1.PNG")
    image2 = cv2.imread(image_save, cv2.IMREAD_UNCHANGED)
    assert (img_transformed_ascend == image2).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=34.0)
def test_dvpp_rotate_operation_02():
    """
    Feature: Rotate operation on device
    Description: Testing the normal functionality of the Rotate operator on device
    Expectation: The Output is equal to the expected output
    """
    # Rotate operator: expand=False
    image = cv2.imread(image_bmp)
    degrees = -100
    resample = mode.Inter.NEAREST
    fill_value = 30
    img_transformed_ascend = vision.Rotate(degrees=degrees, resample=resample, expand=False,
                                           fill_value=fill_value).device("Ascend")(image)
    image_save = os.path.join(TEST_DATA_DATASET_FUNC, "rotate_process", "dvpp_image4.bmp")

    image2 = cv2.imread(image_save, cv2.IMREAD_UNCHANGED)
    assert (img_transformed_ascend == image2).all()

    # Rotate operator: input is jpeg
    image1 = os.path.join(TEST_DATA_DATASET_FUNC, "rotate_process", "dvpp_image.JPEG")
    image_dvpp = cv2.imread(image1)
    image_origin = os.path.join(TEST_DATA_DATASET_FUNC, "rotate_process", "n15075141_1104.JPEG")
    image = cv2.imread(image_origin)

    degrees = 100
    resample = mode.Inter.BILINEAR
    fill_value = 255
    img_transformed_ascend = vision.Rotate(degrees=degrees, resample=resample, expand=True,
                                           fill_value=fill_value).device("Ascend")(image)
    assert np.allclose(img_transformed_ascend, image_dvpp, rtol=39, atol=39)

    # Rotate dvpp operator: pipeline mode
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    degrees = 0
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 0)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    rotate_op_dvpp = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                                   fill_value=fill_value).device("Ascend")
    dataset2 = dataset2.map(input_columns=["image"], operations=rotate_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=rotate_op_dvpp)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image == image_aug).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_rotate_exception_01():
    """
    Feature: Rotate operation on device
    Description: Testing the Rotate Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Rotate dvpp operator, exception test, fill_value equals -1
    degrees = 100
    fill_value = -1
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]."):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, eager mode, 2 channels, dvpp does not support
    degrees = 100
    resample = mode.Inter.NEAREST
    expand = False
    fill_value = (255, 255, 255)
    image = np.random.randn(104, 560, 2).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Rotate(degrees=degrees, resample=resample, expand=expand,
                          fill_value=fill_value).device(device_target="Ascend")(image)
        assert "The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 2" in str(e.value)

    # Rotate dvpp operator, eager mode, unsupported interpolation method BICUBIC
    degrees = 100
    resample = mode.Inter.BICUBIC
    expand = True
    center = (0, 0)
    fill_value = (0, 100, 255)
    image = np.random.randn(24, 56).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                          fill_value=fill_value).device(device_target="Ascend")(image)
        assert "DvppRotate: Invalid Interpolation mode, only support BILINEAR and NEAREST" in str(e.value)

    # Rotate dvpp operator, eager mode, unsupported interpolation method ANTIALIAS
    degrees = 100
    resample = mode.Inter.ANTIALIAS
    expand = True
    center = (0, 0)
    fill_value = (0, 100, 255)
    image = np.random.randn(24, 56).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                          fill_value=fill_value).device(device_target="Ascend")(image)
        assert "DvppRotate: Invalid Interpolation mode, only support BILINEAR and NEAREST" in str(e.value)

    # Rotate dvpp operator, eager mode, unsupported interpolation method PILCUBIC
    degrees = 100
    resample = mode.Inter.PILCUBIC
    expand = True
    center = (0, 0)
    fill_value = (0, 100, 255)
    image = np.random.randn(24, 56).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                          fill_value=fill_value).device(device_target="Ascend")(image)
        assert "DvppRotate: Invalid Interpolation mode, only support BILINEAR and NEAREST" in str(e.value)

    # Rotate dvpp operator, eager mode, unsupported interpolation method AREA
    degrees = 100
    resample = mode.Inter.AREA
    expand = True
    center = (0, 0)
    fill_value = (0, 100, 255)
    image = np.random.randn(24, 56).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                          fill_value=fill_value).device(device_target="Ascend")(image)
        assert "DvppRotate: Invalid Interpolation mode, only support BILINEAR and NEAREST" in str(e.value)

    # Rotate dvpp operator, eager mode, unsupported interpolation method CUBIC
    degrees = 100
    resample = mode.Inter.CUBIC
    expand = True
    center = (0, 0)
    fill_value = (0, 100, 255)
    image = np.random.randn(24, 56).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                          fill_value=fill_value).device(device_target="Ascend")(image)
        assert "DvppRotate: Invalid Interpolation mode, only support BILINEAR and NEAREST" in str(e.value)

    # Rotate dvpp operator, exception test, degrees too large
    degrees = 16777217
    with pytest.raises(ValueError, match="Input degrees is not within the required interval"):
        vision.Rotate(degrees=degrees).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, degrees too small
    degrees = -16777217
    with pytest.raises(ValueError, match="Input degrees is not within the required interval"):
        vision.Rotate(degrees=degrees).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, degrees empty
    degrees = ""
    with pytest.raises(TypeError,
                       match=r'Argument degrees with value "" is not of type \[<class \'float\'>, '
                             r'<class \'int\'>\], but got <class \'str\'>.'):
        vision.Rotate(degrees=degrees).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, degrees is None
    degrees = None
    with pytest.raises(TypeError,
                       match=r"Argument degrees with value None is not of type \[<class \'float\'>, "
                             r"<class \'int\'>\], but got <class \'NoneType\'>."):
        vision.Rotate(degrees=degrees).device(device_target="Ascend")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_dvpp_rotate_exception_02():
    """
    Feature: Rotate operation on device
    Description: Testing the Rotate Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Rotate dvpp operator, exception test, degrees equals list
    degrees = [100, 100]
    with pytest.raises(TypeError,
                       match=r"Argument degrees with value \[100, 100\] is not of type \[<class \'float\'>, "
                             r"<class \'int\'>\], but got <class \'list\'>."):
        vision.Rotate(degrees=degrees).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, resample equals None
    degrees = 100
    resample = None
    with pytest.raises(TypeError,
                       match=r"Argument resample with value None is not of type \[<enum 'Inter'>\], but got <class "
                             r"'NoneType'>."):
        vision.Rotate(degrees=degrees, resample=resample).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, resample equals empty
    resample = ""
    degrees = 7536
    with pytest.raises(TypeError, match=r"Argument resample with value \"\" is not of type \[<enum \'Inter\'>\], "
                                        r"but got <class \'str\'>."):
        vision.Rotate(degrees=degrees, resample=resample).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, expand equals None
    expand = None
    degrees = 100
    with pytest.raises(TypeError,
                       match=r"Argument expand with value None is not of type \[<class 'bool'>\], but got <class "
                             r"'NoneType'>."):
        vision.Rotate(degrees=degrees, expand=expand).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, expand equals empty
    expand = ""
    degrees = 100
    with pytest.raises(TypeError, match=r"Argument expand with value \"\" is not of type \[<class \'bool\'>\], " \
                                        r"but got <class \'str\'>."):
        vision.Rotate(degrees=degrees, expand=expand).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, center equals 3-tuple
    degrees = -100
    center = (100, 200, 300)
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.Rotate(degrees=degrees, center=center).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, center equals empty
    degrees = 200
    center = ""
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.Rotate(degrees=degrees, center=center).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, center equals list
    center = [-100, -200]
    degrees = 100
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.Rotate(degrees=degrees, center=center).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, fill_value equals 256
    fill_value = 256
    degrees = 100
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]"):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, fill_value equals 2-tuple
    degrees = 100
    fill_value = (100, 100)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, fill_value equals empty
    degrees = 100
    fill_value = ""
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, fill_value equals float
    degrees = 100
    fill_value = (10, 100, 25.5)
    with pytest.raises(TypeError, match="value 25.5 is not of type \\[<class 'int'>\\], but got <class 'float'>."):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, fill_value equals 4-tuple
    degrees = 100
    fill_value = (10, 100, 255, 45)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, fill_value equals list
    degrees = 100
    fill_value = [10, 100, 255]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value).device(device_target="Ascend")

    # Rotate dvpp operator, exception test, input equals 0-dimensional
    degrees = 100
    image = 10
    rotate_op = vision.Rotate(degrees=degrees).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        rotate_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_rotate_exception_03():
    """
    Feature: Rotate operation on device
    Description: Testing the Rotate Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Rotate dvpp operator, exception test, no parameters
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.Rotate().device(device_target="Ascend")

    # Rotate dvpp operator, exception test, too many parameters
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 100)
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.Rotate(degrees, resample, expand, center, fill_value, more_para).device(device_target="Ascend")

    # Rotate dvpp operator, input equals list
    degrees = 256
    image = np.random.randn(128, 32, 3)
    rotate_op = vision.Rotate(degrees=degrees).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        rotate_op(list(image))

    # Rotate dvpp operator, input equals tensor
    degrees = 256
    image = np.random.randn(20, 30, 3)
    rotate_op = vision.Rotate(degrees=degrees).device(device_target="Ascend")
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>."):
        rotate_op(ms.Tensor(image))

    # Rotate dvpp operator, input equals tuple
    degrees = 100
    image = np.random.randn(20, 30, 3)
    rotate_op = vision.Rotate(degrees=degrees).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        rotate_op(tuple(image))

    # RotateTest Rotate dvpp op by processing tensor with dim 1
    logger.info("Test Rotate with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    rotate_op = vision.Rotate(90, expand=False).device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="the input tensor is not HW, HWC or 1HWC, but got: 1"):
        rotate_op(input_mindspore)

    # resample Parameter Anomaly Testing
    resample_list = [[], (1), 1, "1", ds]
    for resample in resample_list:
        img = Image.open(IMAGE_FILE)
        with pytest.raises(TypeError, match="Argument resample with value"):
            vision.Rotate(45, resample=resample, expand=False).device(device_target="Ascend")(img)
        img.close()

    # degrees Parameter Anomaly Testing
    degrees_list = [[], {}, (), "1", ds]
    for degrees in degrees_list:
        img = Image.open(IMAGE_FILE)
        with pytest.raises(TypeError, match="Argument degrees with value"):
            vision.Rotate(degrees).device(device_target="Ascend")(img)
        img.close()


if __name__ == '__main__':
    test_dvpp_rotate_operation_01()
    test_dvpp_rotate_operation_02()
    test_dvpp_rotate_exception_01()
    test_dvpp_rotate_exception_02()
    test_dvpp_rotate_exception_03()
