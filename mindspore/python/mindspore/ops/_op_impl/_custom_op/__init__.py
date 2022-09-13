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
from mindspore.ops._op_impl._custom_op.dsd_impl import dsd_matmul
from mindspore.ops._op_impl._custom_op.dsd_back_impl import dsdbpropimpl
from .batchnorm_fold import _batchnorm_fold_tbe
from .batchnorm_fold2 import _batchnorm_fold2_tbe
from .batchnorm_fold2_grad import _batchnorm_fold2_grad_tbe
from .batchnorm_fold2_grad_reduce import _batchnorm_fold2_grad_reduce_tbe
from .batchnorm_fold_grad import _batchnorm_fold_grad_tbe
from .correction_mul import _correction_mul_tbe
from .correction_mul_grad import _correction_mul_grad_tbe
from .fake_learned_scale_quant_perlayer import _fake_learned_scale_quant_perlayer_tbe
from .fake_learned_scale_quant_perlayer_grad import _fake_learned_scale_quant_perlayer_grad_d_tbe
from .fake_learned_scale_quant_perlayer_grad_reduce import _fake_learned_scale_quant_perlayer_grad_d_reduce_tbe
from .fake_learned_scale_quant_perchannel import _fake_learned_scale_quant_perchannel_tbe
from .fake_learned_scale_quant_perchannel_grad import _fake_learned_scale_quant_perchannel_grad_d_tbe
from .fake_learned_scale_quant_perchannel_grad_reduce import _fake_learned_scale_quant_perchannel_grad_d_reduce_tbe
from .fake_quant_perchannel import _fake_quant_perchannel_tbe
from .fake_quant_perchannel_grad import _fake_quant_perchannel_grad_tbe
from .fake_quant_perlayer import _fake_quant_per_layer_tbe
from .fake_quant_perlayer_grad import _fake_quant_per_layer_grad_tbe
from .minmax_update_perchannel import _minmax_update_perchannel_tbe
from .minmax_update_perlayer import _minmax_update_perlayer_tbe
from .matmul_dds_impl import matmul_dds
from .matmul_dds_grad_impl import matmul_dds_grad
