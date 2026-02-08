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
Testing DVPP ResizedCrop operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as t_trans
from mindspore.dataset.vision import Inter as v_Inter
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=30.0)
def test_dvpp_resizedcrop_operation_01():
    """
    Feature: ResizedCrop operation on device
    Description: Testing the normal functionality of the ResizedCrop operator on device
    Expectation: The Output is equal to the expected output
    """
    # Using the resized_crop_dvpp operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    def pyfunc1(img_bytes):
        img_decode = t_trans.Decode().device("Ascend")(img_bytes)
        img_ops = t_trans.ResizedCrop(top=0, left=0, height=6, width=6, size=(100, 50)).device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = t_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    def pyfunc2(img_bytes):
        img_decode = t_trans.Decode()(img_bytes)
        img_ops = t_trans.ResizedCrop(top=0, left=0, height=6, width=6, size=(100, 50))(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = t_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Test ResizedCrop operator: parameter size is 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 6
    resizecrop_op = t_trans.ResizedCrop(top=0, left=0, height=6, width=6, size=size).device(
        device_target="Ascend")
    resizecrop_op_cpu = t_trans.ResizedCrop(top=0, left=0, height=6, width=6, size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=resizecrop_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=resizecrop_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image == image_aug).all()

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.BILINEAR
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.BILINEAR
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.NEAREST
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.NEAREST
    resizecrop_op = t_trans.ResizedCrop(top=5, left=10, height=10, width=6, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=5, left=10, height=10, width=6, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.LINEAR
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.LINEAR
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.CUBIC
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.BICUBIC
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.BICUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_resizedcrop_operation_02():
    """
    Feature: ResizedCrop operation on device
    Description: Testing the normal functionality of the ResizedCrop operator on device
    Expectation: The Output is equal to the expected output
    """
    # Test ResizedCrop operator: image is bmp, interpolation=v_Inter.BICUBIC
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_bmp)
    interpolation = v_Inter.BICUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()

    # Test ResizedCrop operator: image is PNG, interpolation=v_Inter.CUBIC
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(image)
    resizecrop_op_cpu = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation)(image)
    assert (out == resizecrop_op_cpu).all()

    # Test ResizedCrop operator: image is gif, interpolation=v_Inter.CUBIC
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    img_array = np.array(image)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    out = resizecrop_op(img_array)
    out2 = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                               interpolation=interpolation)(img_array)
    assert (out == out2).all()
    image.close()

    # Test ResizedCrop operator: Dimensions containing 1 will be removed from the output results.
    image = np.random.randn(250, 500, 1).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    resizecrop_op(image)
    assert resizecrop_op(image).shape == (10, 10)

    image = np.random.randn(1, 250, 500, 3).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                        interpolation=interpolation).device(device_target="Ascend")
    resizecrop_op(image)
    assert resizecrop_op(image).shape == (10, 10, 3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=10.0)
def test_dvpp_resizedcrop_exception_01():
    """
    Feature: ResizedCrop operation on device
    Description: Testing the ResizedCrop Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Test ResizedCrop operator: image is jpg, v_Inter.ANTIALIAS is not supported.
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.ANTIALIAS
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.AREA
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.AREA
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)

    # Test ResizedCrop operator: image is jpg, interpolation=v_Inter.PILCUBIC
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    interpolation = v_Inter.PILCUBIC
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)

    image = np.random.randn(250, 500, 1).astype(np.uint8)
    interpolation = v_Inter.PILCUBIC
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)

    # Test ResizedCrop operator: image is PNG, interpolation=v_Inter.CUBIC
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)

    interpolation = v_Inter.CUBIC
    with pytest.raises(RuntimeError) as e_log:
        resizecrop_op = t_trans.ResizedCrop(top=0, left=0, height=4, width=3, size=10).device(device_target="Ascend")
        _ = resizecrop_op(image)
        assert "crop size [3 x 4] should not be smaller than [6, 4]" in str(e_log.value)

    with pytest.raises(RuntimeError) as e_log:
        resizecrop_op = t_trans.ResizedCrop(top=0, left=0, height=6, width=4, size=1,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)
        assert "pic width should be in [6, 32768], width is 1." in str(e_log.value)

    image = np.random.randn(32769, 50, 3).astype(np.uint8)
    with pytest.raises(RuntimeError, match="the input shape should be from \\[4, 6\\] to \\[32768, 32768\\]"):
        resizecrop_op = t_trans.ResizedCrop(top=0, left=0, height=10, width=10, size=10,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)

    # Test ResizedCrop operator: not support pil
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    interpolation = v_Inter.CUBIC
    with pytest.raises(TypeError,
                       match="The input PIL Image cannot be executed on Ascend, you can convert "
                             "the input to the numpy ndarray type."):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=10, width=10, size=10,
                                            interpolation=interpolation).device(device_target="Ascend")
        _ = resizecrop_op(image)
        image.close()

    # Test ResizedCrop operator: input data shape is (250, 500, 1), Cropping beyond the boundaries
    image = np.random.randn(250, 500, 1).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    with pytest.raises(RuntimeError,
                       match="DvppResizedCrop: the sum of top and height: 510 exceeds image height: 250"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=500, width=500, size=100,
                                            interpolation=interpolation).device(device_target="Ascend")
        resizecrop_op(image)

    with pytest.raises(RuntimeError,
                       match="DvppResizedCropOp: the output shape should be "
                             "from \\[4, 6\\] to \\[32768, 32768\\], but got \\[1, 1\\]"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=100, width=100, size=1,
                                            interpolation=interpolation).device(device_target="Ascend")
        resizecrop_op(image)

    # Test ResizedCrop operator: input data dtype is np.float64/int64
    image = np.random.randn(250, 500, 1).astype(np.int64)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=500, width=500, size=100,
                                        interpolation=interpolation).device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        resizecrop_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    image = np.random.randn(250, 500, 1).astype(np.float64)
    interpolation = v_Inter.CUBIC
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=500, width=500, size=100,
                                        interpolation=interpolation).device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        resizecrop_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_resizedcrop_exception_02():
    """
    Feature: ResizedCrop operation on device
    Description: Testing the ResizedCrop Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Test ResizedCrop operator: Unsupported shape
    image = np.random.randn(250, 500, 100).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 100"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=50, width=100, size=100,
                                            interpolation=interpolation).device(device_target="Ascend")
        resizecrop_op(image)

    image = np.random.randn(20, 250, 500, 1).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    with pytest.raises(RuntimeError,
                       match="The input tensor NHWC should be 1HWC or HWC"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=50, width=100, size=100,
                                            interpolation=interpolation).device(device_target="Ascend")
        resizecrop_op(image)

    image = np.random.randn(20, ).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    with pytest.raises(RuntimeError,
                       match="DvppResizedCrop: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=50, width=100, size=100,
                                            interpolation=interpolation).device(device_target="Ascend")
        resizecrop_op(image)

    image = np.random.randn(1, 250, 500, 100, 1, 1).astype(np.uint8)
    interpolation = v_Inter.CUBIC
    with pytest.raises(RuntimeError,
                       match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]"):
        resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=50, width=100, size=100,
                                            interpolation=interpolation).device(device_target="Ascend")
        resizecrop_op(image)

    # Test ResizedCrop operator: input passed in is not a PIL or NumPy object
    image = list(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    resizecrop_op = t_trans.ResizedCrop(top=10, left=10, height=50, width=100, size=100).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got \\<class 'list'\\>."):
        resizecrop_op(image)


if __name__ == '__main__':
    test_dvpp_resizedcrop_operation_01()
    test_dvpp_resizedcrop_operation_02()
    test_dvpp_resizedcrop_exception_01()
    test_dvpp_resizedcrop_exception_02()
