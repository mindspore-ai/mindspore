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
Test CutMixBatch in DE
"""
import numpy as np
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision

DATA_DIR = "../data/dataset/testCifar10Data"
DATA_DIR2 = "../data/dataset/testImageNetData2/train/"
DATA_DIR3 = "../data/dataset/testCelebAData/"

GENERATE_GOLDEN = False
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData2", "train/")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testCifar10Data")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    return data_list




















def test_cut_mix_batch_invalid_alpha():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where alpha is invalid
    Expectation: Correct error is raised as expected
    """
    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, -1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)

    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 0.0)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)


def test_cut_mix_batch_invalid_prob():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where prob is invalid
    Expectation: Correct error is raised as expected
    """
    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 1, 2)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)

    with pytest.raises(ValueError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC, 1, -1)
        error_message = "Input is not within the required interval"
        assert error_message in str(error.value)






def test_cut_mix_batch_without_one_hot():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch where labels are not in one-hot format
    Expectation: Correct error is raised as expected
    """
    # CutMixBatch Images
    data1 = ds.Cifar10Dataset(DATA_DIR, num_samples=10, shuffle=False)

    cutmix_batch_op = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)
    data1 = data1.batch(5, drop_remainder=True)
    data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])

    with pytest.raises(RuntimeError) as error:
        images_cutmix = np.array([])
        for idx, (image, _) in enumerate(data1):
            if idx == 0:
                images_cutmix = image.asnumpy()
            else:
                images_cutmix = np.append(images_cutmix, image.asnumpy(), axis=0)
    error_message = "input label is not in shape of <Batch,Class> or <Batch,Row,Class>"
    assert error_message in str(error.value)
    one_hot_suggession = "You may need to perform OneHot and Batch first"
    assert one_hot_suggession in str(error.value)




def test_cut_mix_batch_invalid_label_type():
    """
    Feature: CutMixBatch
    Description: Test CutMixBatch with label in str type
    Expectation: Error is raised as expected
    """
    image = np.random.randint(0, 255, (3, 28, 28, 1), dtype=np.uint8)
    label = np.array([["one"], ["two"], ["three"]])
    with pytest.raises(RuntimeError) as error:
        _ = vision.CutMixBatch(vision.ImageBatchFormat.NHWC)(image, label)
    error_message = "CutMixBatch: invalid label type, label must be in a numeric type"
    assert error_message in str(error.value)
