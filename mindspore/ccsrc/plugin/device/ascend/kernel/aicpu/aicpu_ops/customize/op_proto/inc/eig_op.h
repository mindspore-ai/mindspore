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

#ifndef CUSTOMIZE_OP_PROTO_INC_EIG_OP_H
#define CUSTOMIZE_OP_PROTO_INC_EIG_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the eigenvalue decomposition of a square matrix. \n

* @par Inputs:
* One input, including:
* @li x:A Tensor. Must be one of the following types: float32, float64, complex64, complex128.
 Shape is [N, N]. \n

* @par Attributes:
* @li compute_v: A bool. Indicating whether to compute eigenvectors. \n

* @par Outputs:
* eigen_values: A Tensor. Has the corresponding complex type with "x". Shape is [N, 1].
* eigen_vectors: A Tensor. Has the corresponding complex type with "x". Shape is [N, N] with compute_v true,
 Shape is empty with compute_v false. \n
*/

REG_CUST_OP(Eig)
  .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(eigen_values, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(eigen_vectors, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .ATTR(compute_v, Bool, false)
  .CUST_OP_END_FACTORY_REG(Eig)
}  // namespace ge
#endif