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
"""wukong tiling"""
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingPara
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingStrategy


class WukongTiling(TilingStrategy):
    """A tiling strategy implementation for wukonghuahua model shape"""

    @classmethod
    def strategy_name(cls):
        return "wukong"

    def tiling(self) -> TilingPara:
        """
        反向的空间分布待详细分析
        N = (4096, 1024, 256, 64) 或 77
        Nq = (4096, 1024, 256, 64)
        d = dv = (40, 80, 160,  160)
        """
        if self.N <= 77:  # [77, 64]
            # cross-attention or self-attention of (64, 64, 160)
            self.Bc = self.N
            self.Tc = self.N // self.Bc
            if self.d <= 80:  # [40, 80]
                # 内存瓶颈为在ub中对P*V结果[Br, dv]进行cast
                self.Br = min(self.Nq, 64)
                self.Tr = self.Nq // self.Br
            else:
                self.Br = min(self.Nq, 64)
                self.Tr = self.Nq // self.Br
        else:
            # self-attention
            if self.N == 256:
                self.Bc = 64
                self.Tc = 1
                # 内存瓶颈为在ub中对Q*K的结果[Br, Bc]进行cast
                self.Br = 64
                self.Tr = self.Nq // self.Br
            else:
                self.Bc = 64
                self.Tc = self.N // self.Bc
                self.Br = 64
                self.Tr = self.Nq // self.Br

        self.last_Br = self.Br
        self.last_Bc = self.Bc

        return self.gen_tiling_para()
