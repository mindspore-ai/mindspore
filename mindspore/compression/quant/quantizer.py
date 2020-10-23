# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Base Class of Quantizer."""

from abc import ABC, abstractmethod
from enum import Enum

from ..._checkparam import Validator

__all__ = ["OptimizeOption"]


class OptimizeOption(Enum):
    r"""
    An enum for the model quantization optimize option, currently only support `QAT`.
    """
    # using quantization aware training
    QAT = "QAT"

    def __str__(self):
        return self.value


class Quantizer(ABC):
    """
    Base class of Quantizer. You can implement different kind of quantizer to get different quantization result.

    Notes:
        This class is an abstract class.

    Args:
        optimize_option (OptimizeOption, list or tuple): Specifies the quant algorithm and options. Default:
            OptimizeOption.QAT.
    """
    def __init__(self,
                 optimize_option=OptimizeOption.QAT):
        if not isinstance(optimize_option, list) and not isinstance(optimize_option, tuple):
            optimize_option = [optimize_option]
        for option in optimize_option:
            option = Validator.check_isinstance("optimize_option", option, OptimizeOption)
        self.optimize_option = optimize_option

    @abstractmethod
    def quantize(self, network):
        pass
