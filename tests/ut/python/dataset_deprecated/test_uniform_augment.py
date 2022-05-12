# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test UniformAugment op in Dataset
"""
import pytest

import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as F
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train/"


def test_cpp_uniform_augment_exception_pyops(num_ops=2):
    """
    Feature: UniformAugment Op
    Description: Test C++ op with invalid Python op in transforms list
    Expectation: Invalid input is detected
    """
    logger.info("Test CPP UniformAugment invalid OP exception")

    transforms_ua = [C.RandomCrop(size=[224, 224], padding=[32, 32, 32, 32]),
                     C.RandomHorizontalFlip(),
                     C.RandomVerticalFlip(),
                     C.RandomColorAdjust(),
                     C.RandomRotation(degrees=45),
                     F.Invert()]

    with pytest.raises(TypeError) as e:
        C.UniformAugment(transforms=transforms_ua, num_ops=num_ops)

    logger.info("Got an exception in DE: {}".format(str(e)))
    assert "Type of Transforms[5] must be c_transform" in str(e.value)


if __name__ == "__main__":
    test_cpp_uniform_augment_exception_pyops(num_ops=1)
