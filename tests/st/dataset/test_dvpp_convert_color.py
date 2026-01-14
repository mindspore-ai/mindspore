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
Testing DVPP ConvertColor operation
"""
import os
import pytest
import cv2
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.utils as mode
import mindspore.dataset.vision.transforms as v_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train/")
    data_list.append(data_dir1)
    return data_list


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=38.0)
def test_dvpp_convert_color_operation_01():
    """
    Feature: ConvertColor operation on device
    Description: Testing the normal functionality of the ConvertColor operator on device
    Expectation: The Output is equal to the expected output
    """
    # ConvertColor DVPP operator: Convert BGR images to BGRA images
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA).device(device_target="Ascend")
    dataset1 = dataset1.map(operations=convert, input_columns=["image"])
    dataset2 = dataset2.map(operations=convert_op, input_columns=["image"])
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        cv2.imwrite('ConvertColor_01.jpg', data2["image"])
        cv2.imwrite('ConvertColor_01_cpu.jpg', data1["image"])
        assert (data1["image"] == data2["image"]).all()

    # Using ConvertColor operator in pyfunc
    ms.set_context(device_target="Ascend")

    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=dir_data()[0], shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA).device("Ascend")(img_decode)
        return img_ops

    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA)(img_decode)
        return img_ops

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # ConvertColor DVPP operator: Convert BGR images to BGRA images
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGB images to RGBA images
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2RGBA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2RGBA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert BGRA images to BGR images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2BGR)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2BGR).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGBA images to RGB images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2RGB)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2RGB).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert BGR images to RGBA images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2RGBA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2RGBA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGB images to BGRA images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGRA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGRA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGBA images to BGR images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2BGR)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2BGR).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert BGRA images to RGB images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2RGB)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2RGB).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert BGR images to RGB images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2RGB)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2RGB).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGB images to BGR images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_convert_color_operation_02():
    """
    Feature: ConvertColor operation on device
    Description: Testing the normal functionality of the ConvertColor operator on device
    Expectation: The Output is equal to the expected output
    """
    # ConvertColor DVPP operator: Convert BGRA images to RGBA images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2RGBA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2RGBA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGBA images to BGRA images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2BGRA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2BGRA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert BGR images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2GRAY)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2GRAY).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGB images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2GRAY)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2GRAY).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert GRAY images to BGR images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGR)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGR).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert GRAY images to RGB images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2RGB)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2RGB).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert GRAY images to BGRA images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert GRAY images to RGBA images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2RGBA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2RGBA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert BGRA images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2GRAY)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2GRAY).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Convert RGBA images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2GRAY)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2GRAY).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Test input dimension containing 1
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)(new_arr)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(new_arr)
    assert (convert == convert_op).all()
    assert convert_op.shape == (128, 128, 3)

    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Test input image channel number 1 is normal
    image = np.random.randint(-255, 255, (256, 128, 1)).astype(np.uint8)
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA)(image)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA).device(device_target="Ascend")(image)
    assert (convert == convert_op).all()

    # ConvertColor DVPP operator: Test 4-dimensional image processing normal
    image = np.random.randint(0, 255, (1, 256, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (256, 128, 3))
    convert = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)(new_arr)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(new_arr)
    assert (convert == convert_op).all()
    assert convert_op.shape == (256, 128, 3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_convert_color_exception_01():
    """
    Feature: ConvertColor operation on device
    Description: Testing the ConvertColor Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # ConvertColor DVPP operator: Exception scenario, no convert_mode parameter value passed
    with pytest.raises(TypeError, match=r"missing a required argument: 'convert_mode'"):
        v_trans.ConvertColor().device(device_target="Ascend")

    # ConvertColor DVPP operator: Exception scenario, convert_mode parameter passed as int
    with pytest.raises(TypeError, match=r"Argument convert_mode with value 1 is not of type \[<enum 'ConvertMode'>\],"
                                        r" but got <class 'int'>."):
        v_trans.ConvertColor(1).device(device_target="Ascend")

    # ConvertColor DVPP operator: Exception scenario, convert_mode parameter passed as str
    with pytest.raises(TypeError, match=r"Argument convert_mode with value a is not of type \[<enum 'ConvertMode'>\],"
                                        r" but got <class 'str'>."):
        v_trans.ConvertColor('a').device(device_target="Ascend")

    # ConvertColor DVPP operator: Exception scenario, convert_mode parameter passed as True
    with pytest.raises(TypeError,
                       match=r"Argument convert_mode with value True is not of type \[<enum 'ConvertMode'>\],"
                             r" but got <class 'bool'>."):
        v_trans.ConvertColor(True).device(device_target="Ascend")

    # ConvertColor DVPP operator: Exception scenario, convert_mode parameter passed as None
    with pytest.raises(TypeError,
                       match=r"Argument convert_mode with value None is not of type \[<enum 'ConvertMode'>\], "
                             "but got <class 'NoneType'>."):
        v_trans.ConvertColor(None).device(device_target="Ascend")

    # ConvertColor DVPP operator: Exception scenario, input one-dimensional data
    image = np.random.randint(0, 255, (3)).astype(np.uint8)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
    with pytest.raises(RuntimeError, match=r"DvppConvertColorOp: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        convert_op.device(device_target="Ascend")(image)

    # ConvertColor DVPP operator: Exception scenario, input 4-dimensional data
    image = np.random.randint(0, 255, (10, 10, 10, 3)).astype(np.uint8)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
    with pytest.raises(RuntimeError, match=r"The input tensor NHWC should be 1HWC or HWC"):
        convert_op.device(device_target="Ascend")(image)

    # ConvertColor DVPP operator: Test error when input image array is not uint8 or float32
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float64)
    with pytest.raises(RuntimeError, match="DvppConvertColor: Error"):
        v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(image)

    # ConvertColor DVPP operator: Test error when input image channel number is not 1 or 3
    image = np.random.randint(0, 255, (128, 128, 2)).astype(np.float32)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, 4, but got: 2"):
        v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(image)

    # ConvertColor DVPP operator: Test error when device is not Ascend or CPU
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="Input device_target is not within the valid set of"):
        v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="test")(image)

    # ConvertColor DVPP operator: Test error when 4-dimensional image N is not 1
    image = np.random.randint(0, 255, (2, 256, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC"):
        v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(image)

    # ConvertColor DVPP operator: ConvertColor DVPP operator: Test error when data is 1-dimensional
    image = np.random.randint(0, 255, (256,)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="DvppConvertColorOp: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(image)

    # ConvertColor DVPP operator: Test error when data is 5-dimensional
    image = np.random.randint(0, 255, (1, 256, 128, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor is not of shape"):
        v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR).device(device_target="Ascend")(image)


if __name__ == '__main__':
    test_dvpp_convert_color_operation_01()
    test_dvpp_convert_color_operation_02()
    test_dvpp_convert_color_exception_01()
