# Copyright 2024 Huawei Technologies Co., Ltd
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
This is the test for op input consistency
"""
import copy
import numpy as np
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Border, Inter


def test_transforms_transform_input_consistency():
    """
    Feature: Transforms transform input consistency test
    Description: Check that operator execution does not affect the consistency of input data
    Expectation: Output is equal to the expected output
    """
    seed = ds.config.get_seed()
    ds.config.set_seed(12345)

    def check_result(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert not np.array_equal(output, inputs_copy)

    data = np.array([2.71606445312564e-03, 6.3476562564e-03]).astype(np.float64)
    data_copy = copy.deepcopy(data)
    out = transforms.TypeCast(np.float16)(data_copy)
    check_result(out, data, data_copy)

    ds.config.set_seed(seed)


def test_vision_transform_input_consistency():
    """
    Feature: Vision transform input consistency test
    Description: Check that operator execution does not affect the consistency of input data
    Expectation: Output is equal to the expected output
    """
    seed = ds.config.get_seed()
    ds.config.set_seed(12345)

    data1 = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
    data1_copy = copy.deepcopy(data1)
    data2 = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 2, 3))
    data2_copy = copy.deepcopy(data2)
    data4 = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    data4_copy = copy.deepcopy(data4)
    data5 = Image.open("../data/dataset/apple.jpg")
    data5_copy = copy.deepcopy(data5)

    def check_result_any(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert (output != inputs_copy).any()

    def check_result(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert not np.array_equal(output, inputs_copy)


    data = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    data = data.astype(np.float32)
    data_copy = copy.deepcopy(data)


    out = vision.CenterCrop(1)(data2_copy)
    check_result_any(out, data2, data2_copy)


    assert (data1 == data1_copy).all()

    out = vision.Decode()(data4_copy)
    check_result(out, data4, data4_copy)


    out = vision.HWC2CHW()(data2_copy)
    check_result(out, data2, data2_copy)


    data = np.random.randn(10, 10, 3)
    data_copy = copy.deepcopy(data)

    out = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])(data1_copy)
    check_result_any(out, data1, data1_copy)


    out = vision.RandomColorAdjust(brightness=(0.5, 1), contrast=(0.4, 1), saturation=(0.3, 1))(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.RandomCrop(8, [10, 10, 10, 10], padding_mode=Border.EDGE)(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.RandomCropDecodeResize(size=(500, 520), scale=(0, 10.0), ratio=(0.5, 0.5),
                                        interpolation=Inter.BILINEAR, max_attempts=1)(data4_copy)
    check_result(out, data4, data4_copy)


    data = np.random.randint(254, 255, size=(3, 100, 100)).astype(np.uint8)
    data_copy = copy.deepcopy(data)

    out = vision.RandomHorizontalFlip(1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)


    out = vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5), interpolation=Inter.BILINEAR)(data1_copy)
    check_result(out, data1, data1_copy)


    data = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 3))
    data_copy = copy.deepcopy(data)
    out = vision.RandomVerticalFlip(1.0)(data_copy)
    check_result_any(out, data, data_copy)

    out = vision.Rescale(1.0 / 255.0, -1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.Resize([5, 5], Inter.BICUBIC)(data1_copy)
    check_result(out, data1, data1_copy)


    out = vision.ToNumpy()(data5_copy)
    assert (np.array(data5_copy) == np.array(data5)).all()
    assert (out == np.array(data5_copy)).all()

    out = vision.ToPIL()(data1_copy)
    assert (data1 == data1_copy).all()

    out = vision.ToTensor()(data1_copy)
    check_result(out, data1, data1_copy)

    data = np.array([2.71606445312564e-03, 6.3476562564e-03]).astype(np.float64)
    data_copy = copy.deepcopy(data)
    out = transforms.TypeCast(np.float32)(data_copy)
    check_result_any(out, data, data_copy)


    ds.config.set_seed(seed)
