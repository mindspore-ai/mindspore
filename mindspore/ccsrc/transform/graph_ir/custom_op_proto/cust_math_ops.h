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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_MATH_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(CholeskySolve)
  .INPUT(x1, TensorType({DT_DOUBLE, DT_FLOAT}))
  .INPUT(x2, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(upper, Bool)
  .CUST_OP_END_FACTORY_REG(CholeskySolve)

REG_CUST_OP(Cauchy)
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(size, ListInt)
  .REQUIRED_ATTR(sigma, Float)
  .REQUIRED_ATTR(median, Float)
  .CUST_OP_END_FACTORY_REG(Cauchy)

REG_CUST_OP(CholeskyInverse)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(upper, Bool)
  .CUST_OP_END_FACTORY_REG(CholeskyInverse)

REG_CUST_OP(Eig)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(eigen_values, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .OUTPUT(eigen_vectors, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .REQUIRED_ATTR(compute_v, Bool)
  .CUST_OP_END_FACTORY_REG(Eig)

REG_CUST_OP(Hypot)
  .INPUT(x1, TensorType({DT_DOUBLE, DT_FLOAT}))
  .INPUT(x2, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Hypot)

REG_CUST_OP(MatrixLogarithm)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .CUST_OP_END_FACTORY_REG(MatrixLogarithm)

REG_CUST_OP(Lcm)
  .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
  .CUST_OP_END_FACTORY_REG(Lcm)

REG_CUST_OP(MatrixExp)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(MatrixExp)

REG_CUST_OP(Heaviside)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                        DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(values, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                             DT_UINT32, DT_UINT64, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                         DT_UINT32, DT_UINT64, DT_UINT8}))
  .CUST_OP_END_FACTORY_REG(Heaviside)

REG_CUST_OP(Gcd)
  .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
  .CUST_OP_END_FACTORY_REG(Gcd)

REG_CUST_OP(Orgqr)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .INPUT(tau, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Orgqr)

REG_CUST_OP(TraceGrad)
  .INPUT(y_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(x_shape, TensorType({DT_INT64}))
  .OUTPUT(x_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
          DT_UINT32, DT_UINT64, DT_UINT8}))
  .CUST_OP_END_FACTORY_REG(TraceGrad)

REG_CUST_OP(Lgamma)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(Lgamma)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_MATH_OPS_H_

