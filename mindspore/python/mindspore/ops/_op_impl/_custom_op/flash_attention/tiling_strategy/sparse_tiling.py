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
"""tiling for sparse """
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingPara
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingStrategy


class SparseTiling(TilingStrategy):
    """A tiling strategy implementation for sparse shape"""

    @classmethod
    def strategy_name(cls):
        return "sparse"

    def tiling(self) -> TilingPara:
        self.Br = min(128, self.Nq)
        self.Bc = min(128, self.N)

        self.Tr = self.Nq // self.Br
        self.Tc = self.N // self.Bc

        if self.Nq % self.Br != 0:
            self.last_Br = self.Nq - self.Tr * self.Br
            self.Tr += 1
        else:
            self.last_Br = self.Br
        if self.N % self.Bc != 0:
            self.last_Bc = self.N - self.Tc * self.Bc
            self.Tc += 1
        else:
            self.last_Bc = self.Bc

        return self.gen_tiling_para()
