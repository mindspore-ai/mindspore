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
# ==============================================================================
"""
Test debug hook of debug mode
"""

import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.debug as dbg

# Need to run all these tests in separate processes since
# the global configuration setting of debug_mode may impact other tests running in parallel.
pytestmark = pytest.mark.forked


@pytest.mark.parametrize("debug_mode_flag, debug_hook_list",
                         [(True, [dbg.PrintMetaDataHook()]),
                          (True, [dbg.PrintDataHook()]),
                          (True, [])])
def test_debug_mode_hook(debug_mode_flag, debug_hook_list):
    """
    Feature: Test the debug mode setter function
    Description: Test valid debug hook case for debug mode
    Expectation: Success
    """
    # get original configs to restore after running is done.
    origin_debug_mode = ds.config.get_debug_mode()
    origin_seed = ds.config.get_seed()

    # set debug flag and hook
    ds.config.set_debug_mode(debug_mode_flag=debug_mode_flag, debug_hook_list=debug_hook_list)
    dataset = ds.ImageFolderDataset("../data/dataset/testPK/data", num_samples=5)
    dataset = dataset.map(operations=[vision.Decode(False), vision.CenterCrop((225, 225))])
    for _ in dataset.create_dict_iterator(num_epochs=1):
        pass
    # restore configs
    ds.config.set_debug_mode(origin_debug_mode)
    ds.config.set_seed(origin_seed)


if __name__ == '__main__':
    test_debug_mode_hook(True, [dbg.PrintMetaDataHook()])
    test_debug_mode_hook(True, [dbg.PrintDataHook()])
    test_debug_mode_hook(True, [])
