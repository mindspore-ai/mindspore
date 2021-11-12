# Copyright 2021 Huawei Technologies Co., Ltd
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
Testing Autotune support in DE
"""
import numpy as np
import pytest
import mindspore.dataset as ds


def test_autotune_simple_pipeline():
    """
    Feature: Auto-tuning
    Description: test simple pipeline of autotune - Generator -> Shuffle -> Batch
    Expectation: pipeline runs successfully
    """
    ds.config.set_enable_autotune(True)

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["data"])
    data1 = data1.shuffle(64)
    data1 = data1.batch(32)

    itr = data1.create_dict_iterator(num_epochs=5)
    for _ in range(5):
        for _ in itr:
            pass

    ds.config.set_enable_autotune(False)


def test_autotune_config():
    """
    Feature: Auto-tuning
    Description: test basic config of autotune
    Expectation: config can be set successfully
    """
    autotune_state = ds.config.get_enable_autotune()
    assert autotune_state is False

    ds.config.set_enable_autotune(False)
    autotune_state = ds.config.get_enable_autotune()
    assert autotune_state is False

    with pytest.raises(TypeError):
        ds.config.set_enable_autotune(1)

    autotune_interval = ds.config.get_autotune_interval()
    assert autotune_interval == 100

    ds.config.set_autotune_interval(200)
    autotune_interval = ds.config.get_autotune_interval()
    assert autotune_interval == 200

    with pytest.raises(TypeError):
        ds.config.set_autotune_interval(20.012)

    with pytest.raises(ValueError):
        ds.config.set_autotune_interval(-999)


if __name__ == "__main__":
    test_autotune_simple_pipeline()
    test_autotune_config()
