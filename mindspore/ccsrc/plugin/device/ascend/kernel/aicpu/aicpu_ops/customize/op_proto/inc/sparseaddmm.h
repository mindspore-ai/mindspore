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
* @brief Generate the distribution of dirichlet. \n

* @par Inputs:
* x: A tensor with data type float16 or float. n-D. \n

* @par Attributes:
* @li seed: An optional int. Defaults to 0. \n

* @par Outputs:
* y: A Tensor. Has the type float16 or float. n-D. \n

* @par Third-party framework compatibility
* @li Compatible with the Pytorch operator SampleDirichlet.
*/

REG_CUST_OP(SparseAddmm)
  .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x1_values, TensorType({DT_UINT64, DT_INT64, DT_UINT32, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8,
                                DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(x1_shape, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x2, TensorType({DT_UINT64, DT_INT64, DT_UINT32, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16,
                         DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(x3, TensorType({DT_UINT64, DT_INT64, DT_UINT32, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16,
                         DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(alpha, TensorType({DT_UINT64, DT_INT64, DT_UINT32, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8,
                            DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(beta, TensorType({DT_UINT64, DT_INT64, DT_UINT32, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16,
                           DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(y, TensorType({DT_UINT64, DT_INT64, DT_UINT32, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16,
                         DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(SparseAddmm)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H