/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "transform/graph_ir/op_declare/linalg_ops_declare.h"

#include "ops/arithmetic_op_name.h"
#include "ops/math_ops.h"
#include "ops/arithmetic_ops.h"

namespace mindspore::transform {
// Ger
INPUT_MAP(Ger) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Ger) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Ger) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Ger, kGerOpName, ADPT_DESC(Ger))

// Svd
INPUT_MAP(Svd) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Svd) = {{"compute_uv", ATTR_DESC(compute_uv, AnyTraits<bool>())},
                 {"full_matrices", ATTR_DESC(full_matrices, AnyTraits<bool>())}};
OUTPUT_MAP(Svd) = {{0, OUTPUT_DESC(sigma)}, {1, OUTPUT_DESC(u)}, {2, OUTPUT_DESC(v)}};
REG_ADPT_DESC(Svd, prim::kPrimSvd->name(), ADPT_DESC(Svd))

// LogMatrixDeterminant
INPUT_MAP(LogMatrixDeterminant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LogMatrixDeterminant) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogMatrixDeterminant) = {{0, OUTPUT_DESC(sign)}, {1, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogMatrixDeterminant, kLogMatrixDeterminantOpName, ADPT_DESC(LogMatrixDeterminant))

// MatrixInverse
INPUT_MAP(MatrixInverse) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixInverse) = {{"adjoint", ATTR_DESC(adjoint, AnyTraits<bool>())}};
OUTPUT_MAP(MatrixInverse) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixInverse, kMatrixInverseOpName, ADPT_DESC(MatrixInverse))

// MatrixDeterminant
INPUT_MAP(MatrixDeterminant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixDeterminant) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDeterminant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDeterminant, kMatrixDeterminantOpName, ADPT_DESC(MatrixDeterminant))

// MatrixSolve
INPUT_MAP(MatrixSolve) = {{1, INPUT_DESC(matrix)}, {2, INPUT_DESC(rhs)}};
ATTR_MAP(MatrixSolve) = {{"adjoint", ATTR_DESC(adjoint, AnyTraits<bool>())}};
OUTPUT_MAP(MatrixSolve) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixSolve, kMatrixSolveOpName, ADPT_DESC(MatrixSolve))

// CholeskyGrad
INPUT_MAP(CholeskyGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}};
ATTR_MAP(CholeskyGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(CholeskyGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CholeskyGrad, prim::kPrimCholeskyGrad->name(), ADPT_DESC(CholeskyGrad));

// Geqrf
CUST_INPUT_MAP(Geqrf) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Geqrf) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Geqrf) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(tau)}};
REG_ADPT_DESC(Geqrf, prim::kPrimGeqrf->name(), CUST_ADPT_DESC(Geqrf));

// MatrixTriangularSolve
INPUT_MAP(MatrixTriangularSolve) = {{1, INPUT_DESC(matrix)}, {2, INPUT_DESC(rhs)}};
ATTR_MAP(MatrixTriangularSolve) = {{"lower", ATTR_DESC(lower, AnyTraits<bool>())},
                                   {"adjoint", ATTR_DESC(adjoint, AnyTraits<bool>())}};
OUTPUT_MAP(MatrixTriangularSolve) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixTriangularSolve, prim::kPrimMatrixTriangularSolve->name(), ADPT_DESC(MatrixTriangularSolve));

// LuUnpack
CUST_INPUT_MAP(LuUnpack) = {{1, INPUT_DESC(LU_data)}, {2, INPUT_DESC(LU_pivots)}};
CUST_ATTR_MAP(LuUnpack) = {{"unpack_data", ATTR_DESC(unpack_data, AnyTraits<bool>())},
                           {"unpack_pivots", ATTR_DESC(unpack_pivots, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(LuUnpack) = {{0, OUTPUT_DESC(pivots)}, {1, OUTPUT_DESC(L)}, {2, OUTPUT_DESC(U)}};
REG_ADPT_DESC(LuUnpack, prim::kPrimLuUnpack->name(), CUST_ADPT_DESC(LuUnpack))

// LuUnpackGrad
CUST_INPUT_MAP(LuUnpackGrad) = {{1, INPUT_DESC(L_grad)}, {2, INPUT_DESC(U_grad)}, {3, INPUT_DESC(LU_data)}};
CUST_ATTR_MAP(LuUnpackGrad) = {{"L_grad_flag", ATTR_DESC(L_grad_flag, AnyTraits<bool>())},
                               {"L_grad_flag", ATTR_DESC(L_grad_flag, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(LuUnpackGrad) = {{0, OUTPUT_DESC(L_data_grad)}, {1, OUTPUT_DESC(U_data_grad)}};
REG_ADPT_DESC(LuUnpackGrad, prim::kPrimLuUnpackGrad->name(), CUST_ADPT_DESC(LuUnpackGrad));

// LuSolve
CUST_INPUT_MAP(LuSolve) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(lu_data)}, {3, INPUT_DESC(lu_pivots)}};
CUST_ATTR_MAP(LuSolve) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(LuSolve) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(LuSolve, prim::kPrimLuSolve->name(), CUST_ADPT_DESC(LuSolve));

// Qr
INPUT_MAP(Qr) = {{kIndex1, INPUT_DESC(x)}};
ATTR_MAP(Qr) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(Qr) = {{kIndex2, ATTR_DESC(full_matrices, AnyTraits<bool>())}};
OUTPUT_MAP(Qr) = {{kIndex0, OUTPUT_DESC(q)}, {kIndex1, OUTPUT_DESC(r)}};
REG_ADPT_DESC(Qr, prim::kPrimQr->name(), ADPT_DESC(Qr));

// LinearSumAssignment
CUST_INPUT_MAP(LinearSumAssignment) = {
  {1, INPUT_DESC(cost_matrix)}, {2, INPUT_DESC(dimension_limit)}, {3, INPUT_DESC(maximize)}};
CUST_ATTR_MAP(LinearSumAssignment) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(LinearSumAssignment) = {{0, OUTPUT_DESC(row_ind)}, {1, OUTPUT_DESC(col_ind)}};
REG_ADPT_DESC(LinearSumAssignment, prim::kPrimLinearSumAssignment->name(), CUST_ADPT_DESC(LinearSumAssignment));

// SolveTriangular
CUST_INPUT_MAP(SolveTriangular) = {{1, INPUT_DESC(a)},
                                   {2, INPUT_DESC(b)},
                                   {3, INPUT_DESC(trans)},
                                   {4, INPUT_DESC(lower)},
                                   {5, INPUT_DESC(unit_diagonal)}};
CUST_ATTR_MAP(SolveTriangular) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(SolveTriangular) = {{0, OUTPUT_DESC(x)}};
REG_ADPT_DESC(SolveTriangular, prim::kPrimSolveTriangular->name(), CUST_ADPT_DESC(SolveTriangular));

// SolveTriangularGrad
CUST_INPUT_MAP(SolveTriangularGrad) = {{1, INPUT_DESC(a)},     {2, INPUT_DESC(x)},     {3, INPUT_DESC(dx)},
                                       {4, INPUT_DESC(trans)}, {5, INPUT_DESC(lower)}, {6, INPUT_DESC(unit_diagonal)}};
CUST_ATTR_MAP(SolveTriangularGrad) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(SolveTriangularGrad) = {{0, OUTPUT_DESC(da)}, {1, OUTPUT_DESC(db)}};
REG_ADPT_DESC(SolveTriangularGrad, prim::kPrimSolveTriangularGrad->name(), CUST_ADPT_DESC(SolveTriangularGrad));

}  // namespace mindspore::transform
