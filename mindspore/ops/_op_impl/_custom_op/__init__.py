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
from .batch_matmul_impl import CusBatchMatMul
from .cholesky_trsm_impl import CusCholeskyTrsm
from .fused_abs_max1_impl import CusFusedAbsMax1
from .img2col_impl import CusImg2Col
from .matmul_cube_dense_left_impl import CusMatMulCubeDenseLeft
from .matmul_cube_dense_right_impl import CusMatMulCubeDenseRight
from .matmul_cube_fracz_left_cast_impl import CusMatMulCubeFraczLeftCast
from .matmul_cube_fracz_right_mul_impl import CusMatMulCubeFraczRightMul
from .matmul_cube_impl import CusMatMulCube
from .matrix_combine_impl import CusMatrixCombine
from .transpose02314_impl import CusTranspose02314
