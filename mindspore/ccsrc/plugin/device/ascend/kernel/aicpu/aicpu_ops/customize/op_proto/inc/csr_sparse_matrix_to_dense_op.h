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

#ifndef CUSTOMIZE_OP_PROTO_INC_CSR_SPARSE_MATRIX_TO_DENSE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CSR_SPARSE_MATRIX_TO_DENSE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Converts a (possibly batched) CSRSparseMatrix to dense matrices. \n

* @par Inputs:
* @li x_dense_shape: A vector Tensor of type int32 or int64. 1D.
* @li x_batch_pointers: A vector Tensor of type int32 or int64. 1D.
* @li x_row_pointers: A vector Tensor of type int32 or int64. 1D.
* @li x_values: A matrix Tensor of type float, double, complex64 or complex128. 2D. \n

* @par Outputs:
* @li y: A matrix Tensor. Has the same type as "x_values", with shape "x_dense_shape". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator CSRSparseMatrixToDense.
*/
REG_CUST_OP(CSRSparseMatrixToDense)
  .INPUT(x_dense_shape, TensorType({DT_INT64, DT_INT32}))
  .INPUT(x_batch_pointers, TensorType({DT_INT64, DT_INT32}))
  .INPUT(x_row_pointers, TensorType({DT_INT64, DT_INT32}))
  .INPUT(x_col_indices, TensorType({DT_INT64, DT_INT32}))
  .INPUT(x_values, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(CSRSparseMatrixToDense)
}  // namespace ge
#endif