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
 * @brief: Returns the matrix logarithm of one or more square matrices. \n

 * @par Inputs:
 * @li x: A Tensor. Must be one of the following types:
 *         complex64, complex128. \n

 * @par Outputs:
 * @li y: A Tensor. Has the same type as "x". \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator MaxLogarithm.
 */
REG_CUST_OP(MatrixLogarithm)
  .INPUT(x, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(y, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(MatrixLogarithm)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H