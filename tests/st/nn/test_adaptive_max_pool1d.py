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
# ============================================================================
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_adaptivemaxpool1d_3d(mode):
    """
    Feature: nn.AdaptiveMaxPool1d
    Description: Verify the result of AdaptiveMaxPool1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    a = np.arange(16).reshape(2, 2, 4).astype(np.float32)
    x = Tensor(a)
    except_out_val = np.array([[[1., 3.],
                                [5., 7.]],
                               [[9., 11.],
                                [13., 15.]]], dtype=np.float32)
    out = nn.AdaptiveMaxPool1d(2)(x)
    assert np.allclose(out.asnumpy(), except_out_val)
