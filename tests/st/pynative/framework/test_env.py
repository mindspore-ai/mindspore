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
# ============================================================================

"""Test PyNative Envs"""

import os
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, ops, context, nn
from tests.mark_utils import arg_mark

class EnvContext:
    def __init__(self, enter_callback, exit_callback):
        self.enter_callback = enter_callback
        self.exit_callback = exit_callback

    def __enter__(self):
        self.enter_callback()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_callback()


def test_pynative_disable_auto_h2d():
    """
    Feature: Pynative disable auto h2d copy.
    Description: Test MS_DEV_DISABLE_AUTO_H2H env.
    Expectation: run success
    """
    def set_env():
        os.environ["MS_DEV_DISABLE_AUTO_H2D"] = "1"

    def unset_env():
        os.environ.pop("MS_DEV_DISABLE_AUTO_H2D", None)

    x = Tensor(1.0)
    with EnvContext(set_env, unset_env):
        context.set_context(device_target="Ascend")
        with pytest.raises(RuntimeError):
            y = x + 1
            assert y == 2

        with pytest.raises(RuntimeError):
            y = ops.assign(x, 2)
            assert y == 2

        context.set_context(device_target="CPU")
        y = x + 1
        assert y == 2

        y = ops.assign(x, 2)
        assert y == 2

    x = Tensor(2)
    with EnvContext(set_env, unset_env):
        context.set_context(device_target="Ascend")
        x = x.move_to("Ascend")
        y = x * 2
        assert y == 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_synchronize():
    """
    Feature: Test pynative synchronize
    Description: Test the code for the synchronous branch.
    Expectation: success
    """
    try:
        context.set_context(pynative_synchronize=True)

        # Cell object to be differentiated
        class MulNet(nn.Cell):
            def construct(self, x, y, z):
                return x * y * z

        x = Tensor([1, 2], ms.float32)
        y = Tensor([-2, 3], ms.float32)
        z = Tensor([0, 3], ms.float32)
        net = MulNet()
        net.set_inputs(Tensor(shape=[None], dtype=ms.float32), y, z)
        output = ms.grad(net, grad_position=(1, 2))(x, y, z)
        assert (output[0].asnumpy() == np.array([0, 6], dtype=np.float32)).all()
        assert (output[1].asnumpy() == np.array([-2, 6], dtype=np.float32)).all()
    finally:
        context.set_context(pynative_synchronize=False)
