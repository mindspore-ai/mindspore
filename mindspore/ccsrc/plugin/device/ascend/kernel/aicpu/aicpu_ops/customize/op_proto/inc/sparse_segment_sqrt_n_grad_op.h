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

#ifndef CUSTOMIZE_OP_PROTO_INC_SPARSE_SEGMENT_SQRT_N_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_SPARSE_SEGMENT_SQRT_N_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes gradients for SparseSegmentMean . \n

* @par Inputs:
* The input grad must have be type float or double. Inputs include:
* @li x: A Tensor. Must be one of the following types: float, double.
gradient propagated to the SparseSegmentMean op.
* @li indices: A Tensor. Must be one of the following types: int32, int64.
indices passed to the corresponding SparseSegmentMean op.
* @li segment_ids: A Tensor of type int32. segment_ids passed to the
corresponding SparseSegmentMean op.
* @li output_dim0: A Tensor of type int32. dimension 0 of "x" passed to
SparseSegmentMean op . \n

* @par Outputs:
* y:A Tensor. Has the same type as grad . \n

* @par Third-party framework compatibility
* Compatible with tensorflow SparseSegmentSqrtNGrad operator
*/

REG_CUST_OP(SparseSegmentSqrtNGrad)
  .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16}))
  .INPUT(indices, TensorType({DT_INT32}))
  .INPUT(segment_ids, TensorType({DT_INT32}))
  .INPUT(output_dim0, TensorType({DT_INT32}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(SparseSegmentSqrtNGrad)
}  // namespace ge
#endif