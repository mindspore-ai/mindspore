# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest

from mindspore import context


def test_set_offload_context():
    """
    Feature: configuration of offload .
    Description: offloadContext configuration test case.
    Expectation: assert ok.
    """
    offload_config = {"enable_offload": True, "offload_param": "CPU", "offload_path": "/tmp",
                      "offload_checkpoint": "CPU", "offload_ddr_size": 1024, "offload_disk_size": 4096,
                      "enable_aio": False, "aio_block_size": 1000, "aio_queue_depth": 9999, "enable_pinned_mem": True}
    context.set_offload_context(offload_config=offload_config)
    offload_config_ = context.get_offload_context()
    enable_offload = offload_config_.get("enable_offload", None)
    offload_param = offload_config_.get("offload_param", None)
    offload_path = offload_config_.get("offload_path", None)
    offload_checkpoint = offload_config_.get("offload_checkpoint", None)
    offload_ddr_size = offload_config_.get("offload_ddr_size", None)
    offload_disk_size = offload_config_.get("offload_disk_size", None)
    enable_aio = offload_config_.get("enable_aio", None)
    aio_block_size = offload_config_.get("aio_block_size", None)
    aio_queue_depth = offload_config_.get("aio_queue_depth", None)
    enable_pinned_mem = offload_config_.get("enable_pinned_mem", None)
    assert enable_offload
    assert offload_param == "cpu"
    assert offload_path == "/tmp"
    assert offload_checkpoint == "cpu"
    assert offload_ddr_size == 1024
    assert offload_disk_size == 4096
    assert not enable_aio
    assert aio_block_size == 1000
    assert aio_queue_depth == 9999
    assert enable_pinned_mem
    with pytest.raises(ValueError):
        context.set_offload_context(offload_config={"offload_param": "gpu"})
