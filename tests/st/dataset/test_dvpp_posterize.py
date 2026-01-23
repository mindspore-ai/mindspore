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
Testing DVPP Posterize operation
"""
import os
import pytest
import numpy as np
import cv2
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=102.0)
def test_dvpp_posterize_operation_01():
    """
    Feature: Posterize operation on device
    Description: Testing the normal functionality of the Posterize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Testing the normal pipeline mode
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    posterize_op = v_trans.Posterize(bits=8)
    posterize_op_dvpp = v_trans.Posterize(bits=8).device(device_target="Ascend")
    dataset2 = dataset2.map(input_columns=["image"], operations=posterize_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=posterize_op_dvpp)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image == image_aug).all()

    # Using the Posterize operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.Posterize(bits=8).device("Ascend")(
            img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.Posterize(bits=8)(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Test posterize dvpp func jpg image bits=0
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    posterize_op = v_trans.Posterize(bits=0)(image)
    posterize_op_dvpp = v_trans.Posterize(bits=0).device(device_target="Ascend")(image)
    assert (posterize_op == posterize_op_dvpp).all()

    # Test posterize dvpp func jpg image bits=8
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    posterize_op = v_trans.Posterize(bits=8)(image)
    posterize_op_dvpp = v_trans.Posterize(bits=8).device(device_target="Ascend")(image)
    assert (posterize_op == posterize_op_dvpp).all()

    # Test posterize dvpp func bmp image bits=6
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_bmp)
    posterize_op = v_trans.Posterize(bits=6)(image)
    posterize_op_dvpp = v_trans.Posterize(bits=6).device(device_target="Ascend")(image)
    assert (posterize_op == posterize_op_dvpp).all()

    # Test posterize dvpp func png image bits=3
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    posterize_op = v_trans.Posterize(bits=3)(image)
    posterize_op_dvpp = v_trans.Posterize(bits=3).device(device_target="Ascend")(image)
    assert (posterize_op == posterize_op_dvpp).all()

    # Test posterize dvpp func gif image bits=7
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_gif)
    img_array = np.array(image)
    posterize_op = v_trans.Posterize(bits=7)(img_array)
    posterize_op_dvpp = v_trans.Posterize(bits=7).device(device_target="Ascend")(img_array)
    assert (posterize_op == posterize_op_dvpp).all()
    image.close()

    # Test posterize dvpp func input with numpy 1 channel
    image = np.random.randn(24, 56, 1).astype(np.uint8)
    new_arr = np.reshape(image, (24, 56))
    posterize_op = v_trans.Posterize(bits=7)(new_arr)
    posterize_op_dvpp = v_trans.Posterize(bits=7).device(device_target="Ascend")(image)
    assert (posterize_op == posterize_op_dvpp).all()
    assert posterize_op_dvpp.shape == (24, 56)

    # Test posterize dvpp func input with numpy 3channel
    image = np.random.randn(1, 24, 56, 3).astype(np.uint8)
    new_arr = np.reshape(image, (24, 56, 3))
    posterize_op = v_trans.Posterize(bits=2)(new_arr)
    posterize_op_dvpp = v_trans.Posterize(bits=2).device(device_target="Ascend")(image)
    assert (posterize_op == posterize_op_dvpp).all()
    assert posterize_op_dvpp.shape == (24, 56, 3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=130.0)
def test_dvpp_posterize_exception_01():
    """
    Feature: Posterize operation on device
    Description: Testing the Posterize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Test posterize dvpp func channel 2
    image = np.random.randn(24, 56, 2).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 2"):
        _ = v_trans.Posterize(bits=4).device(device_target="Ascend")(image)

    # Test posterize dvpp func dtype float32
    image = np.random.randn(24, 56, 3).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        _ = v_trans.Posterize(bits=4).device(device_target="Ascend")(image)
        assert "The input data is not uint8" in str(e.value)

    # Test posterize dvpp func error_input_size
    image = np.random.randn(3, 3).astype(np.float32)
    with pytest.raises(RuntimeError,
                       match="DvppPosterizeOp: the input shape should be from \\[4, 6\\] to "
                             "\\[8192, 4096\\], but got \\[3, 3\\]"):
        _ = v_trans.Posterize(bits=4).device(device_target="Ascend")(image)

    image = np.random.randn(8193, 4097).astype(np.float32)
    with pytest.raises(RuntimeError,
                       match="DvppPosterizeOp: the input shape should be from \\[4, 6\\] to "
                             "\\[8192, 4096\\], but got \\[8193, 4097\\]"):
        _ = v_trans.Posterize(bits=4).device(device_target="Ascend")(image)

    # Test posterize dvpp func error_bits_type
    image = np.random.randn(24, 56, 3).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Argument bits with value.* is not of type \\[<class 'int'>\\],"
                             " but got <class 'tuple'>"):
        _ = v_trans.Posterize(bits=(1, 2)).device(device_target="Ascend")(image)

    # Test posterize dvpp func error_bits_type
    image = np.random.randn(24, 56, 3).astype(np.uint8)
    with pytest.raises(TypeError,
                       match="Argument bits with value 1.5 is not of type \\[<class 'int'>\\], "
                             "but got <class 'float'>"):
        _ = v_trans.Posterize(bits=1.5).device(device_target="Ascend")(image)

    # Test posterize dvpp func error_bits_type
    image = np.random.randn(24, 56, 3).astype(np.uint8)
    with pytest.raises(ValueError,
                       match="nput bits is not within the required interval of \\[0, 8\\]"):
        _ = v_trans.Posterize(bits=-5).device(device_target="Ascend")(image)


if __name__ == '__main__':
    test_dvpp_posterize_operation_01()
    test_dvpp_posterize_exception_01()
