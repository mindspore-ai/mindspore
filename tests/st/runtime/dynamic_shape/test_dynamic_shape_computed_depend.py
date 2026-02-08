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

import os
import argparse
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, jit
from mindspore import dtype as mstype


class UniqueNet(nn.Cell):
    def __init__(self):
        super(UniqueNet, self).__init__()
        self.unique = P.Unique()
        self.square = P.Square()
        self.relu = P.ReLU()

    @jit(backend="ms_backend")
    def construct(self, x):
        x, _ = self.unique(x)
        x = self.square(x)
        return self.relu(x)


def test_computed_depend_case_with_conf_thread_num():
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.float32)
    net = UniqueNet()
    output = net(x)
    expect = np.array([1, 4, 9])
    assert (output.asnumpy() == expect).all()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="test_computed_depend_case_with_conf_thread_num")
        parser.add_argument("--thread_num", type=int, default=5, help="thread number")
        parser.add_argument("--enable_pipeline", type=bool, default=True, help="enable pipeline")
        parser.add_argument("--enable_new_pipeline", type=bool, default=True, help="enable new pipeline")
        args_opt = parser.parse_args()

        pipeline_env = ""
        if not args_opt.enable_pipeline:
            pipeline_env = "pipeline:False"
        elif not args_opt.enable_new_pipeline:
            pipeline_env = "new_pipeline:False"
        os.environ["MS_DEV_RUNTIME_CONF"] = pipeline_env
        ms.runtime.dispatch_threads_num(args_opt.thread_num)

        test_computed_depend_case_with_conf_thread_num()
    finally:
        os.unsetenv("MS_DEV_RUNTIME_CONF")
