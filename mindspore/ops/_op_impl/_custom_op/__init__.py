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

"""custom ops"""
from .fake_learned_scale_quant_perlayer import _fake_learned_scale_quant_perlayer_tbe
from .fake_learned_scale_quant_perlayer_grad import _fake_learned_scale_quant_perlayer_grad_d_tbe
from .fake_learned_scale_quant_perlayer_grad_reduce import _fake_learned_scale_quant_perlayer_grad_d_reduce_tbe
from .fake_learned_scale_quant_perchannel import _fake_learned_scale_quant_perchannel_tbe
from .fake_learned_scale_quant_perchannel_grad import _fake_learned_scale_quant_perchannel_grad_d_tbe
from .fake_learned_scale_quant_perchannel_grad_reduce import _fake_learned_scale_quant_perchannel_grad_d_reduce_tbe
