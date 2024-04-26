/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_LINALG_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_LINALG_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(Geqrf)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(tau, TensorType({DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Geqrf)

REG_CUST_OP(LuUnpack)
    .INPUT(LU_data, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(LU_pivots, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(pivots, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(L, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(U, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(unpack_data, Bool, true)
    .ATTR(unpack_pivots, Bool, true)
    .CUST_OP_END_FACTORY_REG(LuUnpack)

REG_CUST_OP(LuUnpackGrad)
  .INPUT(L_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .INPUT(U_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .INPUT(LU_data, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .OUTPUT(L_data_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .OUTPUT(U_data_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .REQUIRED_ATTR(L_grad_flag, Bool)
  .CUST_OP_END_FACTORY_REG(LuUnpackGrad)

REG_CUST_OP(LuSolve)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(lu_data, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(lu_pivots, TensorType({DT_INT32}))
  .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(LuSolve)

REG_CUST_OP(LinearSumAssignment)
  .INPUT(cost_matrix, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                                  DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(dimension_limit, TensorType({DT_INT64}))
  .INPUT(maximize, TensorType({DT_BOOL}))
  .OUTPUT(row_ind, TensorType({DT_INT64}))
  .OUTPUT(col_ind, TensorType({DT_INT64}))
  .CUST_OP_END_FACTORY_REG(LinearSumAssignment)

REG_CUST_OP(SolveTriangular)
  .INPUT(a, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_COMPLEX64,
                        DT_COMPLEX128}))
  .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_COMPLEX64,
                        DT_COMPLEX128}))
  .INPUT(trans, TensorType({DT_INT64}))
  .INPUT(lower, TensorType({DT_BOOL}))
  .INPUT(unit_diagonal, TensorType({DT_BOOL}))
  .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(SolveTriangular)

REG_CUST_OP(SolveTriangularGrad)
  .INPUT(a, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_COMPLEX128, DT_COMPLEX64, DT_INT16, DT_INT32, DT_INT64,
                        DT_INT8}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_COMPLEX128, DT_COMPLEX64}))
  .INPUT(dx, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_COMPLEX128, DT_COMPLEX64}))
  .INPUT(trans, TensorType({DT_INT64}))
  .INPUT(lower, TensorType({DT_BOOL}))
  .INPUT(unit_diagonal, TensorType({DT_BOOL}))
  .OUTPUT(da, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(db, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(SolveTriangularGrad)

REG_CUST_OP(LstsqV2)
  .INPUT(a, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .INPUT(b, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .OPTIONAL_INPUT(driver, TensorType({DT_INT64}))
  .OUTPUT(solution, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(residuals, TensorType({DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(rank, TensorType({DT_INT64}))
  .OUTPUT(singular_values, TensorType({DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(LstsqV2)

REG_CUST_OP(LstsqV2Grad)
  .INPUT(gx, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .INPUT(a, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .INPUT(b, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .OUTPUT(ga, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(gb, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(LstsqV2Grad)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_LINALG_OPS_H_
