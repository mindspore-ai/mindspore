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
Testing RandomPerspective op in DE
"""

import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"






def test_random_perspective_exception_distortion_scale_range():
    """
    Feature: RandomPerspective op
    Description: Test RandomPerspective where distortion_scale is not in [0, 1]
    Expectation: Error is raised as expected
    """
    logger.info("test_random_perspective_exception_distortion_scale_range")
    try:
        _ = vision.RandomPerspective(distortion_scale=1.5)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input distortion_scale is not within the required interval of [0.0, 1.0]."


def test_random_perspective_exception_prob_range():
    """
    Feature: RandomPerspective op
    Description: Test RandomPerspective where prob is not in [0, 1]
    Expectation: Error is raised as expected
    """
    logger.info("test_random_perspective_exception_prob_range")
    try:
        _ = vision.RandomPerspective(prob=1.2)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input prob is not within the required interval of [0.0, 1.0]."
