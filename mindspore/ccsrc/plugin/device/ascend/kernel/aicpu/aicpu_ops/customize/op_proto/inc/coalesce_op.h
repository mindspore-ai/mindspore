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

#ifndef CUSTOMIZE_OP_PROTO_INC_COALESCE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_COALESCE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Returns the coalesced sparse tensor of the input. \n

* @par Inputs:
* Three inputs, contains:
* @li x_indices: A 2-D `Tensor` of type `int64`. It's elements should be non-negative.
* The `indices` of nonzero elements of the `SparseTensor`, size `[ndims, nnz]`. \n
* @li x_values: A 1-D `Tensor`. Must be one of the following types: float32, float16.
* The `values` of the `SparseTensor`, size `[nnz]`. \n
* @li x_shape: A 1-D `Tensor` of type `int64`.
* The `shape` of the `SparseTensor`, size `[ndims]`. \n

* @par Outputs:
* @li y_indices: A 2-D `Tensor` of type `int64`. It's elements are non-negative.
* The `indices` of nonzero elements of the coalesced `SparseTensor`, size `[ndims, nnz_coalesced]`.
* The `nnz_coalesced` represents the number of different indices in `x_indices` \n
* @li y_values: A 1-D `Tensor`. Has the same type as `x_values`.
* The `values` of the coalesced `SparseTensor`, size `[nnz_coalesced]`. \n
* @li y_shape: A 1-D `Tensor` of type `int64`.
* The `shape` of the coalesced `SparseTensor`, size `[ndims]`. \n

* @par Third-party framework compatibility.
* Compatible with the Pytorch operator Coalesce. \n
*/
REG_CUST_OP(Coalesce)
  .INPUT(x_indices, TensorType({DT_INT64}))
  .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(x_shape, TensorType({DT_INT64}))
  .OUTPUT(y_indices, TensorType({DT_INT64}))
  .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y_shape, TensorType({DT_INT64}))
  .CUST_OP_END_FACTORY_REG(Coalesce)
}  // namespace ge
#endif