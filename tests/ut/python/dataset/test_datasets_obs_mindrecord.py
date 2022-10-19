# Copyright 2022 Huawei Technologies Co., Ltd
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
Test OBSMindDataset operations
"""
import pytest

from mindspore.dataset.engine.datasets_standard_format import OBSMindDataset
from mindspore import log as logger

DATA_DIR = ["s3://dataset/imagenet0", "s3://dataset/imagenet1"]


def test_obs_mindrecord_exception():
    """
    Feature: Test OBSMindDataset.
    Description: Invalid input.
    Expectation: Raise exception.
    """

    logger.info("Test error cases for MnistDataset")
    error_msg_0 = "Argument dataset_files"
    with pytest.raises(TypeError, match=error_msg_0):
        OBSMindDataset("err_dataset", "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_0_1 = "Item of dataset files"
    with pytest.raises(TypeError, match=error_msg_0_1):
        OBSMindDataset([1, 2], "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_1 = "Argument server"
    with pytest.raises(TypeError, match=error_msg_1):
        OBSMindDataset(DATA_DIR, 12, "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_1_1 = "server should"
    with pytest.raises(ValueError, match=error_msg_1_1):
        OBSMindDataset(DATA_DIR, "ftp://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_2 = "Argument ak"
    with pytest.raises(TypeError, match=error_msg_2):
        OBSMindDataset(DATA_DIR, "https://dummy_site", 12, "dummy_sk", "s3://dummy_sync_dir")

    error_msg_3 = "Argument sk"
    with pytest.raises(TypeError, match=error_msg_3):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak", 12, "s3://dummy_sync_dir")

    error_msg_4 = "Argument sync_obs_path"
    with pytest.raises(TypeError, match=error_msg_4):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak", "dummy_sk", 12)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", num_shards=4, shard_id=4)
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", num_shards=2, shard_id=4)

    error_msg_7 = "Argument shard_equal_rows"
    with pytest.raises(TypeError, match=error_msg_7):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", shard_equal_rows=1)


if __name__ == '__main__':
    test_obs_mindrecord_exception()
