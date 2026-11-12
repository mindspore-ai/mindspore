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
Testing the MixUpBatch op in DE
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testCifar10Data"
DATA_DIR2 = "../data/dataset/testImageNetData2/train/"
DATA_DIR3 = "../data/dataset/testCelebAData/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False
























def test_mixup_batch_fail5():
    """
    Feature: MixUpBatch op
    Description: Test MixUpBatch op where labels are not OneHot encoded
    Expectation: Error is raised as expected
    """
    logger.info("test_mixup_batch_fail5")

    # Original Images
    ds_original = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)
    ds_original = ds_original.batch(5)

    images_original = np.array([])
    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original, image.asnumpy(), axis=0)

    # MixUp Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    mixup_batch_op = vision.MixUpBatch()
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=mixup_batch_op, input_columns=["image", "label"])

    with pytest.raises(RuntimeError) as error:
        images_mixup = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_mixup = image.asnumpy()
            else:
                images_mixup = np.append(images_mixup, image.asnumpy(), axis=0)
    error_message = "wrong labels shape. The second column (labels) must have a shape of NC or NLC"
    assert error_message in str(error.value)


def test_mix_up_batch_invalid_label_type():
    """
    Feature: MixUpBatch
    Description: Test MixUpBatch with label in str type
    Expectation: Error is raised as expected
    """
    image = np.random.randint(0, 255, (3, 28, 28, 1), dtype=np.uint8)
    label = np.array([["one"], ["two"], ["three"]])
    with pytest.raises(RuntimeError) as error:
        _ = vision.MixUpBatch()(image, label)
    error_message = "MixUpBatch: invalid label type, label must be in a numeric type"
    assert error_message in str(error.value)




def test_mixup_exception_01():
    """
    Feature: MixUp operation
    Description: Testing the MixUp Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # MixUp Operator: Exception Testing, alpha=0.0
    with pytest.raises(ValueError, match="Input alpha must be greater than 0"):
        vision.MixUp(batch_size=3, alpha=0.0, is_single=False)

    # MixUp Operator: Exception Testing, alpha=-0.5
    with pytest.raises(ValueError, match="Input alpha must be greater than 0"):
        vision.MixUp(batch_size=3, alpha=-0.5)

    # MixUp Operator: Exception Testing, alpha=0
    with pytest.raises(ValueError, match="Input alpha must be greater than 0"):
        vision.MixUp(batch_size=3, alpha=0, is_single=False)

    # MixUp Operator: Exception Testing, alpha= 'afd ^%$ 123'
    alpha = "afd ^%$ 123"
    with pytest.raises(TypeError, match="123 is not of type \\[<class 'int'>, <class 'float'>\\]."):
        vision.MixUp(batch_size=3, alpha=alpha)

    # MixUp Operator: Exception Testing, The batch_size parameter is missing.
    with pytest.raises(TypeError, match="missing a required argument: 'batch_size'"):
        vision.MixUp(alpha=1.0, is_single=True)

    # MixUp Operator: Exception Testing, The alpha parameter is missing.
    with pytest.raises(TypeError, match="missing a required argument: 'alpha'"):
        vision.MixUp(batch_size=3, is_single=True)

    # MixUp Operator: Exception Testing, No parameters passed
    with pytest.raises(TypeError, match="missing a required argument: 'batch_size'"):
        vision.MixUp()

    # MixUp Operator: Exception Testing, batch_size = 0
    with pytest.raises(ValueError, match=r"Input is not within the required interval of \[1, 16777216\]."):
        vision.MixUp(batch_size=0, alpha=1.0, is_single=False)

    # MixUp Operator: Exception Testing, batch_size = 16777217
    with pytest.raises(ValueError, match=r"Input is not within the required interval of \[1, 16777216\]."):
        vision.MixUp(batch_size=16777217, alpha=1.0, is_single=False)

    # MixUp Operator: Normal testing, is_single is of type int
    with pytest.raises(TypeError, match=r'Argument is_single with value 1 is not of type \[\<class \'bool\'\>\].'):
        _ = vision.MixUp(batch_size=16, alpha=True, is_single=1)

    # MixUp Operator: Normal testing, is_single is of type str
    with pytest.raises(TypeError, match=r'Argument is_single with value true is not of type \[\<class \'bool\'\>\].'):
        _ = vision.MixUp(batch_size=16, alpha=1, is_single='true')

    # MixUp Operator: Normal testing, batch_size is of type bool
    with pytest.raises(TypeError, match="Argument batch_size with value True is not "
                                        "of type \\(<class 'int'>,\\), but got <class 'bool'>."):
        vision.MixUp(batch_size=True, alpha=0.2, is_single=False)

    # MixUp Operator: Normal testing, batch_size is of type str
    with pytest.raises(TypeError, match="Argument batch_size with value true is not of type \\[<class "
                                        "'int'>\\], but got <class 'str'>."):
        vision.MixUp(batch_size='true', alpha=0.2, is_single=False)

    # MixUp Operator: Normal testing, batch_size is of type float
    with pytest.raises(TypeError, match="Argument batch_size with value 0.1 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'float'>."):
        _ = vision.MixUp(batch_size=0.1, alpha=0.2, is_single=False)

    # MixUp Operator: Normal testing, alpha is of type bool
    with pytest.raises(TypeError, match="Argument alpha with value True is not of type \\(<class 'int'>, "
                                        "<class 'float'>\\), but got <class 'bool'>."):
        _ = vision.MixUp(batch_size=16, alpha=True, is_single=False)
