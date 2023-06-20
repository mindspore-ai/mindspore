/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Solves a linear system of equations with a positive semidefinite matrix
  to be inverted given its Cholesky factor matrix u . \n

* @par Inputs:
* x1:A Tensor. input matrix b of size [..., M, K], where ... is zero or more batch dimensions.
* x2:A Tensor. input matrix u of size [..., M, M], where ... is zero of more batch dimensions
  composed of upper or lower triangular Cholesky factor . \n

* @par Attributes:
* upper:An optional bool. Defaults to False.Boolean indicating whether to
  consider the Cholesky factor as a lower or upper triangular matrix . \n

* @par Outputs:
* y:A Tensor. Has the same type as x1 . \n

* @attention Constraints:
* The input x2 is a tensor of shape [..., M, M] whose inner-most 2 dimensions
  form square matrices.

* @par Third-party framework compatibility
* Compatible with pytorch cholesky_solve operator.
*/
REG_CUST_OP(CholeskySolve)
  .INPUT(x1, TensorType::RealNumberType())
  .INPUT(x2, TensorType::RealNumberType())
  .OUTPUT(y, TensorType::RealNumberType())
  .ATTR(upper, Bool, false)
  .CUST_OP_END_FACTORY_REG(CholeskySolve)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H