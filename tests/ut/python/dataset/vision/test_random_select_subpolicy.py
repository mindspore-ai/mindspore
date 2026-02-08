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
Test RandomSelectSubpolicy op in Dataset
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision as visions
from util import config_get_set_seed

TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def test_random_select_subpolicy():
    """
    Feature: RandomSelectSubpolicy Op
    Description: Test C++ implementation, both valid and invalid input
    Expectation: Dataset pipeline runs successfully and results are verified for valid input.
        Invalid input is detected.
    """
    original_seed = config_get_set_seed(0)

    def test_config(arr, policy):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=visions.RandomSelectSubpolicy(policy), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # 3 possible outcomes
    policy1 = [[(ops.PadEnd([4], 0), 0.5), (ops.Compose([ops.Duplicate(), ops.Concatenate()]), 1)],
               [(ops.Slice([0, 1]), 0.5), (ops.Duplicate(), 1), (ops.Concatenate(), 1)]]
    res1 = test_config([[1, 2, 3]], policy1)
    assert res1 in [[[1, 2, 1, 2]], [[1, 2, 3, 1, 2, 3]], [[1, 2, 3, 0, 1, 2, 3, 0]]]

    # test exceptions
    assert "policy can not be empty." in test_config([[1, 2, 3]], [])
    assert "policy[0] can not be empty." in test_config([[1, 2, 3]], [[]])
    assert "op of (op, prob) in policy[1][0] is neither a transforms op (TensorOperation) nor a callable pyfunc" \
           in test_config([[1, 2, 3]], [[(ops.PadEnd([4], 0), 0.5)], [(1, 0.4)]])
    assert "prob of (op, prob) policy[1][0] is not within the required interval of [0, 1]" in test_config([[1]], [
        [(ops.Duplicate(), 0)], [(ops.Duplicate(), -0.1)]])

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_random_select_subpolicy_operation_01():
    """
    Feature: RandomSelectSubpolicy operation
    Description: Testing the normal functionality of the RandomSelectSubpolicy operator
    Expectation: The Output is equal to the expected output
    """
    # test randomselectsubpolicy prob = 1
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    policy = [
        [(visions.RandomRotation((45, 45)), 1)]
    ]
    operations = visions.RandomSelectSubpolicy(policy=policy)
    dataset2 = dataset2.map(input_columns=["image"], operations=operations)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # test randomselectsubpolicy all para
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    policy = [
        [(visions.RandomRotation((45, 45)), 0.5), (visions.RandomVerticalFlip(), 1),
         (visions.RandomColorAdjust(), 0.8)],
        [(visions.RandomRotation((90, 90)), 1), (visions.RandomColorAdjust(), 0.2)]
    ]
    operations = visions.RandomSelectSubpolicy(policy=policy)
    dataset2 = dataset2.map(input_columns=["image"], operations=operations)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # test randomselectsubpolicy jpg
    image = Image.open(image_jpg)
    policy = [
        [(visions.AutoContrast(), 0.5),
         (visions.CenterCrop(200), 1),
         (visions.Invert(), 0)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)

    # test randomselectsubpolicy bmp
    image = Image.open(image_bmp)
    policy = [
        [(visions.RandomRotation((45, 45)), 0.01), (visions.RandomVerticalFlip(), 0.01),
         (visions.RandomColorAdjust(), 0.01)],
        [(visions.RandomRotation((90, 90)), 1), (visions.RandomColorAdjust(), 1)],
        [(visions.AutoContrast(), 0.5), (visions.CenterCrop((200, 200)), 0.01), (visions.Invert(), 0)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)

    # test randomselectsubpolicy png
    image = Image.open(image_png).convert("RGB")
    policy = [
        [(visions.RandomRotation(125), 0.5)],
        [(visions.RandomHorizontalFlip(0.9), 1), (visions.RandomCrop(150, 150), 0.2)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)

    # test randomselectsubpolicy gif
    image = Image.open(image_gif)
    policy = [
        [(visions.RandomRotation(125), 1)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)

    # test randomselectsubpolicy jpg
    image = Image.open(image_jpg)
    policy = [
        [(visions.Resize(20), 1)],
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)

    # test randomselectsubpolicy image.shape=(256, 360, 3)
    image = np.random.randn(256, 360, 3).astype(np.uint8)
    policy = [
        [(visions.Invert(), 1), (visions.RandomColorAdjust(), 1), (visions.RandomCrop((150, 150)), 1)],
        [(visions.RandomCrop((150, 150)), 1), (visions.RandomRotation((30, 60)), 1)],
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)


def test_random_select_subpolicy_operation_02():
    """
    Feature: RandomSelectSubpolicy operation
    Description: Testing the normal functionality of the RandomSelectSubpolicy operator
    Expectation: The Output is equal to the expected output
    """
    # test randomselectsubpolicy image.shape=(256, 360, 1)
    image = np.random.randint(0, 255, (256, 360, 1)).astype(np.uint8)
    policy = [
        [(visions.AutoContrast(), 0.5), (visions.CenterCrop(size=(250, 252)), 0.6)],
        [(visions.RandomCrop((150, 150)), 0.3), (visions.RandomHorizontalFlip(0.9), 0.3)],
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    random_select_subpolicy_op(image)

    # test randomselectsubpolicy image.shape=(1000, 1000)
    image = np.random.randint(0, 255, (1000, 1000)).astype(np.uint8)
    policy = [
        [(visions.AutoContrast(), 0.685)],
        [(visions.RandomCrop((150, 150)), 0.2), (visions.RandomHorizontalFlip(0.9), 0.3)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    _ = random_select_subpolicy_op(image)


def test_random_select_subpolicy_exception_01():
    """
    Feature: RandomSelectSubpolicy operation
    Description: Testing the RandomSelectSubpolicy Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test randomselectsubpolicy no v_c_trans op
    policy = [
        [(1, 0.5)]
    ]
    with pytest.raises(TypeError, match="op of \\(op, prob\\) in policy\\[0\\]\\[0\\] is neither "
                                        "a transforms op \\(TensorOperation\\) nor a callable pyfunc."):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy prob = 1.1
    policy = [
        [(visions.RandomRotation((45, 45)), 1.1)]
    ]
    with pytest.raises(ValueError, match="Input prob of \\(op, prob\\) policy\\[0\\]\\[0\\] "
                                         "is not within the required interval of \\[0, 1\\]."):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy prob = -0.1
    policy = [
        [(visions.RandomRotation((45, 45)), -0.1)]
    ]
    with pytest.raises(ValueError, match="not within the required interval"):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy prob =
    policy = [
        [(visions.RandomRotation((45, 45)), "")]
    ]
    with pytest.raises(TypeError, match="not supported"):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy no prob
    policy = [
        [(visions.RandomRotation((45, 45)))]
    ]
    with pytest.raises(ValueError, match="needs to be a 2-tuple."):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy policy = [()]
    policy = [
        ((visions.RandomRotation((45, 45))),)
    ]
    with pytest.raises(TypeError, match="Argument policy"):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy policy = []
    policy = []
    with pytest.raises(ValueError, match="policy can not be empty."):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy policy = ""
    policy = ""
    with pytest.raises(TypeError, match="Argument policy"):
        visions.RandomSelectSubpolicy(policy=policy)

    # test randomselectsubpolicy no para
    with pytest.raises(TypeError, match="missing a required argument"):
        visions.RandomSelectSubpolicy()

    # test randomselectsubpolicy no para
    policy = [
        [(visions.RandomRotation((45, 45)), 0.5), (visions.RandomVerticalFlip(), 1),
         (visions.RandomColorAdjust(), 0.8)],
        [(visions.RandomRotation((90, 90)), 1), (visions.RandomColorAdjust(), 0.2)]
    ]
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        visions.RandomSelectSubpolicy(policy, more_para)

    # test randomselectsubpolicy Invert input is error
    image = np.random.randint(0, 255, (256, 360)).astype(np.uint8)
    policy = [
        [(visions.Invert(), 1)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    with pytest.raises(RuntimeError, match="input tensor is not in shape of <H,W,C>, but got rank: 2"):
        random_select_subpolicy_op(image)

    # test randomselectsubpolicy input is error
    image = np.random.randint(0, 255, (256, 360, 3, 3)).astype(np.uint8)
    policy = [
        [(visions.Invert(), 1)]
    ]
    random_select_subpolicy_op = visions.RandomSelectSubpolicy(policy)
    with pytest.raises(RuntimeError, match="input tensor is not in shape of <H,W,C>"):
        random_select_subpolicy_op(image)

    # test randomselectsubpolicy py AutoContrast
    policy = [
        [(visions.AutoContrast(), 1)]
    ]
    visions.RandomSelectSubpolicy(policy)

    # test randomselectsubpolicy str
    policy = [
        [(visions.AutoContrast(), "0.1")]
    ]
    with pytest.raises(TypeError, match="not supported between instances of 'str' and 'int'"):
        visions.RandomSelectSubpolicy(policy)


def test_random_select_subpolicy_exception_02():
    """
    Feature: RandomSelectSubpolicy operation
    Description: Testing the RandomSelectSubpolicy Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test randomselectsubpolicy list
    policy = [
        [(visions.AutoContrast(), [0.5, 0.6])]
    ]
    with pytest.raises(TypeError, match="not supported between instances of 'list' and 'int'"):
        visions.RandomSelectSubpolicy(policy)

    # test randomselectsubpolicy all is list
    policy = [
        [[visions.AutoContrast(), 0.5]]
    ]
    with pytest.raises(ValueError, match="Value policy\\[0\\]\\[0\\] needs to be a 2-tuple."):
        visions.RandomSelectSubpolicy(policy)

    # test randomselectsubpolicy two tuple
    policy = [
        ((visions.AutoContrast(), 0.5),)
    ]
    with pytest.raises(TypeError, match="is not of type \\[<class 'list'>\\]."):
        visions.RandomSelectSubpolicy(policy)

    # test randomselectsubpolicy policy is np
    policy = np.array([[(visions.AutoContrast(), 0.5)]])
    with pytest.raises(TypeError, match="is not of type \\[<class 'list'>\\]."):
        visions.RandomSelectSubpolicy(policy)

    # test randomselectsubpolicy 0.6
    policy = [[(0.6, visions.AutoContrast())]]
    with pytest.raises(TypeError, match="op of \\(op, prob\\) in policy\\[0\\]\\[0\\] is neither "
                                        "a transforms op \\(TensorOperation\\) nor a callable pyfunc."):
        visions.RandomSelectSubpolicy(policy)


if __name__ == "__main__":
    test_random_select_subpolicy()
    test_random_select_subpolicy_operation_01()
    test_random_select_subpolicy_operation_02()
    test_random_select_subpolicy_exception_01()
    test_random_select_subpolicy_exception_02()
