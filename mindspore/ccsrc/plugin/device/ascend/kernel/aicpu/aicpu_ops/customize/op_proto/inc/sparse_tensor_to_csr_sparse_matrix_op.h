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

#ifndef CUSTOMIZE_OP_PROTO_INC_SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_OP_H
#define CUSTOMIZE_OP_PROTO_INC_SPARSE_TENSOR_TO_CSR_SPARSE_MATRIX_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
 * @brief Converts a (possibly batched) SparseTensor to a CSRSparseMatrix . \n

 * @par Inputs:
 * @li x_indices: A 2D Tensor of type int32 or int64.
 * @li x_values: A 1D Tensor of type float, double, complex64 or complex128.
 * @li x_dense_shape: A 1D Tensor of type int32 or int64. The shape of the dense output tensor . \n

 * @par Outputs:
 * @li y_dense_shape: A 1D Tensor of the same type as "x_dense_shape".
 * @li y_batch_pointers: A 1D Tensor of the same type as "x_indices".
 * @li y_row_pointers: A 1D Tensor of the same type as "x_indices".
 * @li y_col_indices: A 1D Tensor of the same type as "x_indices".
 * @li y_values: A 1D Tensor of the same type as "x_values" . \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator SparseTensorToCSRSparseMatrix.
 */
REG_CUST_OP(SparseTensorToCSRSparseMatrix)
  .INPUT(x_indices, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x_values, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(x_dense_shape, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_dense_shape, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_batch_pointers, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_row_pointers, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_col_indices, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_values, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(SparseTensorToCSRSparseMatrix)
}  // namespace ge
#endif