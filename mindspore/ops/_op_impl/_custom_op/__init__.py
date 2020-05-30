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
from .batchnorm_fold import _batchnorm_fold_tbe
from .batchnorm_fold_grad import _batchnorm_fold_grad_tbe
from .batchnorm_fold2 import _batchnorm_fold2_tbe
from .batchnorm_fold2_grad import _batchnorm_fold2_grad_tbe
from .batchnorm_fold2_grad_reduce import _batchnorm_fold2_grad_reduce_tbe
from .correction_mul import _correction_mul_tbe
from .correction_mul_grad import _correction_mul_grad_tbe
from .fake_quant_with_min_max import _fake_quant_tbe
from .fake_quant_with_min_max_grad import _fake_quant_grad_tbe
from .fake_quant_with_min_max_update import _fake_quant_update5d_tbe