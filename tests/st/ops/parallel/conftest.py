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
"""Pytest configuration for parallel ops tests."""
import mindspore as ms
import pytest


@pytest.fixture(params=[
    (ms.GRAPH_MODE, 'O0'),
    (ms.PYNATIVE_MODE, ''),
], autouse=True)
def _init(request):
    mode, jit_level = request.param
    ms.communication.init()
    ms.set_context(mode=mode, device_target='Ascend')
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': jit_level})

    yield

    ms.communication.release()
