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
"""The base class of tiling strategy"""
from abc import ABCMeta
from abc import abstractmethod
from collections import namedtuple

TilingPara = namedtuple("TilingPara", "Br last_Br Bc last_Bc Tr Tc")


class TilingStrategy(metaclass=ABCMeta):
    """Tiling strategy interface. All implementations should be defined in this module,
    otherwise, the UT will fail.
    """

    _strategies = {}

    def __init__(self, Nq, N, head_dim) -> None:
        super().__init__()
        self.Nq = Nq
        self.N = N
        self.Br = None
        self.last_Br = None
        self.Bc = None
        self.last_Bc = None
        self.Tr = None
        self.Tc = None
        self.d = head_dim

    def __init_subclass__(cls, **kwargs):
        TilingStrategy._strategies[cls.strategy_name()] = cls

    @classmethod
    @abstractmethod
    def strategy_name(cls):
        """strategy name"""
        raise NotImplementedError

    @classmethod
    def from_strategy_name(cls, stgy_name: str):
        """from strategy name"""
        stgy_clz = TilingStrategy._strategies.get(stgy_name)
        if stgy_clz is None:
            raise Exception(f"Strategy:{stgy_name} not supported")

        return stgy_clz

    @abstractmethod
    def tiling(self) -> TilingPara:
        """tiling"""
        raise NotImplementedError

    def gen_tiling_para(self) -> TilingPara:
        """gen tiling para"""
        return TilingPara(self.Br, self.last_Br, self.Bc, self.last_Bc, self.Tr, self.Tc)
