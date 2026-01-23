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
Testing DVPP VerticalFlip operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


DATA_DIR = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=64.0)
def test_dvpp_vertical_flip_operation_01():
    """
    Feature: VerticalFlip operation on device
    Description: Testing the normal functionality of the VerticalFlip operator on device
    Expectation: The Output is equal to the expected output
    """
    # VerticalFlip Operator: Pipeline Mode
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    verticalflip_op_cpu = v_trans.VerticalFlip()
    dataset2 = dataset2.map(input_columns=["image"], operations=verticalflip_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=verticalflip_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image == image_aug).all()

    # VerticalFlip Operator: Pipeline Mode, Flip Two Images
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    dataset2 = dataset2.map(input_columns=["image"], operations=verticalflip_op)
    dataset2 = dataset2.padded_batch(2, pad_info={"image": ([None, None, 3], 0)})
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Using the VerticalFlip operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.VerticalFlip().device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.VerticalFlip()(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec)(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # VerticalFlip Operator: Eager mode, input is a PNG image
    image = cv2.imread(image_png)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    out = verticalflip_op(image)
    out2 = np.flipud(image)
    assert (out == out2).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_vertical_flip_operation_02():
    """
    Feature: VerticalFlip operation on device
    Description: Testing the normal functionality of the VerticalFlip operator on device
    Expectation: The Output is equal to the expected output
    """
    # VerticalFlip Operator: Eager mode, input is a GIF image
    image = Image.open(image_gif)
    img_array = np.array(image)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    out = verticalflip_op(img_array)
    out2 = np.flipud(img_array)
    assert (out == out2).all()
    image.close()

    # VerticalFlip Operator: Eager mode, input is a JPG image
    image = cv2.imread(image_jpg)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    out = verticalflip_op(image)
    out2 = np.flipud(image)
    assert (out == out2).all()

    # VerticalFlip Operator: Eager mode, numpy shape=20x30x3, input image data channel count: 1 or 3.
    image = np.random.randint(0, 255, (20, 30, 3)).astype(np.uint8)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    out = verticalflip_op(image)
    out2 = np.flipud(image)
    assert (out == out2).all()

    image = np.random.randint(0, 255, (20, 30, 1)).astype(np.float32)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    out = verticalflip_op(image)
    assert out.shape == (20, 30)

    # VerticalFlip Operator:  Test 4D
    image_reshape = np.random.randint(0, 255, (1, 30, 8, 3)).astype(np.uint8)
    input_4_shape = image_reshape.shape
    num_batch = input_4_shape[0]
    out_4_list = []
    batch_1d = 0
    while batch_1d < num_batch:
        out_4_list.append(cv2.flip(image_reshape[batch_1d], 0))
        batch_1d += 1
    out_4_cv = np.array(out_4_list).astype(np.uint8)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    out = verticalflip_op(image_reshape)
    assert (out == out_4_cv).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_dvpp_vertical_flip_exception_01():
    """
    Feature: VerticalFlip operation on device
    Description: Testing the VerticalFlip Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # VerticalFlip Operator: not support pil
    image = Image.open(image_gif)
    with pytest.raises(TypeError,
                       match="The input PIL Image cannot be executed on Ascend, you can convert "
                             "the input to the numpy ndarray type."):
        verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_op(image)
        image.close()

    # VerticalFlip Operator: numpy shape=20x30x8
    image = np.random.randint(0, 255, (20, 30, 8)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[H,W,C\] is not 1, 3, but got: 8"):
        verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_op(image)

    # verticalflip operator: Test image is pic width should be in [6, 4096]
    input_4_dim = np.random.randint(0, 255, (1, 3, 3, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError) as e_log:
        verticalflip_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_flip_op(input_4_dim)
        assert "pic width should be in \\[6, 4096\\], width is 3." in str(e_log.value)

    input_3_dim = np.random.randint(0, 255, (3, 4097, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError) as e_log:
        verticalflip_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_flip_op(input_3_dim)
        assert "pic width should be in \\[6, 4096\\], width is 4097." in str(e_log.value)

    input_3_dim = np.random.randint(0, 255, (8193, 100, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError) as e_log:
        verticalflip_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_flip_op(input_3_dim)
        assert "the input shape should be from \\[4, 6\\] to \\[8192, 4096\\]." in str(e_log.value)

    # VerticalFlip Operator:  test 5d 6d 7d
    input_5_dim = np.random.randint(0, 255, (2, 3, 6, 7, 8)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]."):
        verticalflip_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_flip_op(input_5_dim)

    input_6_dim = np.random.randint(0, 255, (2, 3, 6, 7, 8, 1)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]."):
        verticalflip_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_flip_op(input_6_dim)

    input_7_dim = np.random.randint(0, 255, (2, 3, 6, 7, 8, 1, 1)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]."):
        verticalflip_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = verticalflip_flip_op(input_7_dim)

    # VerticalFlip Operator:  Test int64/float64
    image = np.random.randint(0, 255, (20, 30, 8)).astype(np.int64)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        verticalflip_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    image = np.random.randint(0, 255, (20, 30, 8)).astype(np.float64)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        verticalflip_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    # VerticalFlip Operator: Test 1D
    image = np.random.randint(0, 255, (20,)).astype(np.uint8)
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        verticalflip_op(image)
        assert "DvppVerticalFlip: invalid input shape, only support NHWC input, got rank: 1." in str(e_log.value)

    # VerticalFlip Operator: input is list
    image = list(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got \\<class 'list'\\>."):
        verticalflip_op(image)

    # VerticalFlip Operator: input is 2D
    image = tuple(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got \\<class 'tuple'\\>."):
        verticalflip_op(image)

    # VerticalFlip Operator: Input parameters
    with pytest.raises(TypeError, match="positional argument but 2 were given"):
        v_trans.VerticalFlip(1).device(device_target="Ascend")

    # VerticalFlip Operator: input is int
    image = 10
    verticalflip_op = v_trans.VerticalFlip().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got \\<class 'int'\\>."):
        verticalflip_op(image)

    # verticalflip operator: When the input image is 4-dimensional, NHWC requires N to be 1.
    input_4_dim = np.random.randint(0, 255, (3, 3, 3, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC."):
        horizontal_flip_op = v_trans.VerticalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_4_dim)


if __name__ == '__main__':
    test_dvpp_vertical_flip_operation_01()
    test_dvpp_vertical_flip_operation_02()
    test_dvpp_vertical_flip_exception_01()
