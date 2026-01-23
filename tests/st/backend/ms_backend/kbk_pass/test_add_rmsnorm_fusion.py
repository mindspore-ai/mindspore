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
# ==============================================================================
import numpy as np
from tests.mark_utils import arg_mark
from tests.st.backend.ms_backend.kbk_pass.util import Capture, capture
import mindspore as ms
from mindspore import nn, Tensor, Parameter, ops

class AddRmsNormNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gamma = Parameter(Tensor(np.ones([2]).astype(np.float32)), name="gamma")
        self.add = ops.Add()
        self.rms_norm = ops.RmsNorm()

    def construct(self, x1, x2):
        return self.rms_norm(self.add(x1, x2), self.gamma)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
def test_add_rms_norm_fusion():
    """
    Feature: AddRmsNormFusion.
    Description: Test pass AddRmsNormFusion.
    Expectation: No exception and got expected AddRmsNorm.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    cap = Capture('add_rms_norm_fusion', 'AddRmsNorm')
    with capture(cap):
        net = AddRmsNormNet()
        x1 = Tensor(np.ones([4, 2]), dtype=ms.float32)
        x2 = Tensor(np.ones([4, 2]), dtype=ms.float32)
        expect = np.ones([4, 2]).astype(np.float32) / 1.000001
        output = net(x1, x2)
        assert np.allclose(output[0].asnumpy(), expect, 1.0e-5, 1.0e-5)

    patterns = ['Default/AddRmsNorm-op']
    cap.check_output(patterns)
