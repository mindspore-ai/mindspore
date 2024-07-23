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

import shutil
import os
from mindspore import context


class AssertGKEnable:
    def __init__(self, enable_graph_kernel=True):
        self.enable_gk = enable_graph_kernel
        self.ir_path = ""

    @staticmethod
    def _rm_dir(dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

    def __enter__(self):
        if self.enable_gk:
            self.ir_path = os.path.join("./irs_{}".format(os.getpid()))
            context.set_context(save_graphs=True, save_graphs_path=self.ir_path)

    def __exit__(self, *args):
        if self.enable_gk:
            graph_kernel_ir_dir = os.path.join(self.ir_path, "verbose_ir_files/graph_kernel")
            if not os.path.isdir(graph_kernel_ir_dir) or not os.listdir(graph_kernel_ir_dir):
                self._rm_dir(self.ir_path)
                raise RuntimeError("Graph Kernel Fusion is not enabled")
            self._rm_dir(self.ir_path)
