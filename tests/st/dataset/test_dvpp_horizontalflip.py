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
Testing DVPP HorizontalFlip operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.transforms.transforms as t_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=100.0)
def test_dvpp_horizontal_flip_operation_01():
    """
    Feature: HorizontalFlip operation on device
    Description: Testing the normal functionality of the HorizontalFlip operator on device
    Expectation: The Output is equal to the expected output
    """
    # Using the HorizontalFlip operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase: map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.HorizontalFlip().device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.HorizontalFlip()(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # HorizontalFlip operator: Test image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    out = horizontal_flip_op(image)
    out2 = np.fliplr(image)
    assert (out == out2).all()

    # HorizontalFlip operator: Test image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    img_array = np.array(image)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    out = horizontal_flip_op(img_array)
    out2 = np.fliplr(img_array)
    assert (out == out2).all()
    image.close()

    # HorizontalFlip operator: Test image is jpg
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    out = horizontal_flip_op(image)
    out2 = np.fliplr(image)
    assert (out == out2).all()

    # HorizontalFlip operator: Test shape is (20, 30, 3),The number of channels in the image data is either 1 or 3.
    image = np.random.randint(0, 255, (20, 30, 3)).astype(np.float32)
    verticalflip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    out = verticalflip_op(image)
    out2 = np.fliplr(image)
    assert (out == out2).all()

    image = np.random.randint(0, 255, (20, 30, 1)).astype(np.uint8)
    verticalflip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    out = verticalflip_op(image)
    assert out.shape == (20, 30)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_horizontal_flip_operation_02():
    """
    Feature: HorizontalFlip operation on device
    Description: Testing the normal functionality of the HorizontalFlip operator on device
    Expectation: The Output is equal to the expected output
    """
    # HorizontalFlip operator: Test PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "")
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        v_trans.Decode(to_pil=True),
        v_trans.HorizontalFlip().device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # HorizontalFlip operator: Test image is 4d numpy data
    input_4_dim = np.random.randint(0, 255, (1, 6, 6, 3)).astype(np.uint8)
    input_4_shape = input_4_dim.shape
    num_batch = input_4_shape[0]

    out_4_list = []
    batch_1d = 0
    while batch_1d < num_batch:
        out_4_list.append(np.flip(input_4_dim[batch_1d], 1))
        batch_1d += 1

    out_4_np = np.array(out_4_list).astype(np.uint8)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    out_4_mindspore = horizontal_flip_op(input_4_dim)

    assert (out_4_np == out_4_mindspore).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_horizontal_flip_exception_01():
    """
    Feature: HorizontalFlip operation on device
    Description: Testing the HorizontalFlip Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # The test input data is a PIL image.
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    with pytest.raises(TypeError,
                       match="The input PIL Image cannot be executed on Ascend, you can convert "
                             "the input to the numpy ndarray type."):
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(image)
        image.close()

    # HorizontalFlip operator: Test shape is (20, 30, 8),The number of channels in the image data is either 1 or 3.
    image = np.random.randint(0, 255, (20, 30, 8)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[H,W,C\] is not 1, 3, but got: 8"):
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(image)

    # HorizontalFlip operator: Test image is pic width should be in [6, 4096]
    input_4_dim = np.random.randint(0, 255, (1, 3, 3, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError) as e_log:
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_4_dim)
        assert r"pic width should be in \[6, 4096\], width is 3." in str(e_log.value)

    input_3_dim = np.random.randint(0, 255, (3, 4097, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError) as e_log:
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_3_dim)
        assert r"pic width should be in \[6, 4096\], width is 4097." in str(e_log.value)

    # HorizontalFlip operator: When the input image is 4-dimensional, NHWC requires N to be 1.
    input_4_dim = np.random.randint(0, 255, (3, 3, 3, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC."):
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_4_dim)

    # HorizontalFlip operator: Test image is int64,float64
    image = np.random.randint(0, 255, (20, 30, 3)).astype(np.int64)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        _ = horizontal_flip_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    image = np.random.randint(0, 255, (20, 30, 3)).astype(np.float64)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        _ = horizontal_flip_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    # HorizontalFlip operator: Test image is 1d
    image = np.random.randint(0, 255, (20,)).astype(np.uint8)
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e_log:
        horizontal_flip_op(image)
    assert "DvppHorizontalFlip: invalid input shape, only support NHWC input, got rank: 1" in str(e_log.value)

    # HorizontalFlip operator: Test image is 2d list
    image = list(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        out = horizontal_flip_op(image)
        out2 = np.fliplr(image)
        assert (out == out2).all()

    # HorizontalFlip operator: Test image is 2d tuple
    image = tuple(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        out = horizontal_flip_op(image)
        out2 = np.fliplr(image)
        assert (out == out2).all()

    # HorizontalFlip operator: Test more arguments
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'test'"):
        v_trans.HorizontalFlip(test='test').device(device_target="Ascend")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_horizontal_flip_exception_02():
    """
    Feature: HorizontalFlip operation on device
    Description: Testing the HorizontalFlip Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Test input data is 5d 6d 7d
    input_5_dim = np.random.randint(0, 255, (2, 1, 6, 7, 8)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]."):
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_5_dim)

    input_6_dim = np.random.randint(0, 255, (2, 1, 6, 7, 8, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]."):
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_6_dim)

    input_7_dim = np.random.randint(0, 255, (2, 1, 6, 7, 8, 3, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match=r"The input tensor is not of shape \[H,W\], \[H,W,C\] or \[N,H,W,C\]."):
        horizontal_flip_op = v_trans.HorizontalFlip().device(device_target="Ascend")
        _ = horizontal_flip_op(input_7_dim)


if __name__ == '__main__':
    test_dvpp_horizontal_flip_operation_01()
    test_dvpp_horizontal_flip_operation_02()
    test_dvpp_horizontal_flip_exception_01()
    test_dvpp_horizontal_flip_exception_02()
