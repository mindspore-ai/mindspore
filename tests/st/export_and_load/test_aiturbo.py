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
# ============================================================================
import os
import pytest
import numpy as np

from mindspore.train import ModelCheckpoint, CheckpointConfig
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_aiturbo_save_error():
    '''
    Feature: aiturbo raise import error
    Description: save checkpoint without aiturbo
    Expectation: success
    '''
    os.environ["AITURBO"] = "1"
    with pytest.raises(ImportError):
        ModelCheckpoint()
    if "AITURBO" in os.environ:
        del os.environ["AITURBO"]


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_aiturbo_CheckpointConfig():
    '''
    Feature: CheckpointConfig init
    Description: CheckpointConfig init with enable_redundance
    Expectation: success
    '''
    config = CheckpointConfig(save_checkpoint_seconds=100, keep_checkpoint_per_n_minutes=5, enable_redundance=True)
    assert np.allclose(config.enable_redundance, True)
