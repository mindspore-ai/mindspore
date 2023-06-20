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

#ifndef CUSTOMIZE_OP_PROTO_INC_SPARSE_SEGMENT_SQRT_N_WITH_NUM_SEGMENTS_OP_H
#define CUSTOMIZE_OP_PROTO_INC_SPARSE_SEGMENT_SQRT_N_WITH_NUM_SEGMENTS_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the sum along sparse segments of a tensor divided by the sqrt of N. \n

* @par Inputs:
* The input indices and segment_ids must have same rank. Inputs include:
* @li x: A Tensor. Must be one of the following types: float16, float32, double.
* @li indices: A Tensor. Must be one of the following types: int32, int64.
A 1-D tensor. Has same rank as segment_ids.
* @li segment_ids: A Tensor. Must be one of the following types: int32, int64.
A 1-D tensor. Values should be sorted and can be repeated .
* @li num_segments:A Tensor or a scalar. Must be one of the following types: int32, int64.
The Tensor must be a 1-D tensor. Values should be sorted and can be repeated. \n

* @par Outputs:
* y:A Tensor. Has the same type as x . \n

* @par Third-party framework compatibility
* Compatible with tensorflow SparseSegmentSqrtNWithNumSegments operator
*/

REG_CUST_OP(SparseSegmentSqrtNWithNumSegments)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
  .INPUT(segment_ids, TensorType({DT_INT32, DT_INT64}))
  .INPUT(num_segments, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(SparseSegmentSqrtNWithNumSegments)
}  // namespace ge
#endif