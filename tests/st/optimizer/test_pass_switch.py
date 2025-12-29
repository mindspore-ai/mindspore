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
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, context
from mindspore import ops
import uuid
from pathlib import Path
import shutil

from tests.mark_utils import arg_mark

class Add_RmsNorm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.RmsNorm = ops.RmsNorm(epsilon=1e-5)

    def construct(self, x1, x2, gamma):
        res = x1 + x2
        hidden_states, _ = self.RmsNorm(res, gamma)
        return hidden_states, res


def _test_add_rmsnorm_fusion():
    dtype = "float16"
    shape = (1, 1024, 11264)

    np.random.seed(0)
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=0, device_target="Ascend",
                        enable_graph_kernel=False)
    context.set_context(
        jit_config={"jit_level": "O0", "infer_boost": "on"})

    np_dtype_map = {"float16": np.float16,
                    "bfloat16": np.float32,
                    "float32": np.float32}
    ms_dtype_map = {"float16": ms.float16,
                    "bfloat16": ms.bfloat16,
                    "float32": ms.float32}
    np_dtype = np_dtype_map[dtype]
    tensor_dtype = ms_dtype_map[dtype]
    gamma_dtype = np_dtype
    gamma_ms_dtype = tensor_dtype

    input_x = np.random.rand(*shape).astype(np_dtype)
    input_y = np.random.rand(*shape).astype(np_dtype)
    gamma = np.ones([shape[-1]]).astype(gamma_dtype)
    net = Add_RmsNorm()

    net(Tensor(input_x, dtype=tensor_dtype), Tensor(input_y, dtype=tensor_dtype),
        Tensor(gamma, dtype=gamma_ms_dtype))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pass_switch_by_add_rms_norm():
    """
    Feature: test pass switch in graph mode
    Description: test pass switch.
    Expectation: run pass.
    """

    def _check_ir(path, pattern, expect_exist):
        res = list(path.rglob(pattern))
        shutil.rmtree(path.name)
        if res and not expect_exist:
            raise RuntimeError("IR should be %s!" % ("existed" if expect_exist else "not existed"))

    ori_env = None
    if os.environ.get("MS_DISABLE_INTERNAL_KERNELS_LIST"):
        ori_env = os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"]

    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "AddRmsNorm"
    irs_path = Path(f"pass_switch_{uuid.uuid4().hex}")
    ms.set_context(save_graphs=3, save_graphs_path=irs_path.name)
    _test_add_rmsnorm_fusion()
    _check_ir(irs_path, "*add_rms_norm_fusion*.ir", True)

    ms.set_context(graph_kernel_flags="--disable_pass=add_rms_norm_fusion")
    _test_add_rmsnorm_fusion()
    _check_ir(irs_path, "*add_rms_norm_fusion*.ir", False)

    if ori_env is None:
        os.environ.pop("MS_DISABLE_INTERNAL_KERNELS_LIST")
    else:
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ori_env
