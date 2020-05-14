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
"""
quantization.

User can use aware quantization to train a model. Mindspore supports quantization aware training,
which models quantization errors in both the forward and backward passes using fake-quantization
ops. Note that the entire computation is carried out in floating point. At the end of quantization
aware training, Mindspore provides conversion functions to convert the trained model into lower precision.
"""

from .quant import convert_quant_network

__all__ = ["convert_quant_network"]
