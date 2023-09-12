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

#ifndef CUSTOMIZE_OP_PROTO_INC_MEDIAN_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MEDIAN_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
 * @brief Returns the median of the values in input.
 *
 * @par Inputs
 * one input including:
 * @li x: input A Tensor.Must be one of the RealNumberType types.
 *
 * @par Attributes:
 * @li global_median: whether the output is the global median of all elements or just in the dim.
 * @li axis: the dimension to reduce.
 * @li keepdim: whether the output tensor has dim retained or not.
 *
 * @par Output:
 * one output including:
 * @li y: The output format is (Tensor, Tensor),The first tensor will be populated with the median values and the
 * second tensor, which must have dtype long, with their indices in the dimension dim of input.
 */
REG_CUST_OP(Median)
  .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT16, DT_INT32, DT_INT64}))
  .OUTPUT(values, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT16, DT_INT32, DT_INT64}))
  .OUTPUT(indices, TensorType({DT_INT32, DT_INT64}))
  .REQUIRED_ATTR(global_median, Bool)
  .ATTR(axis, Int, 0)
  .ATTR(keepdim, Bool, false)
  .CUST_OP_END_FACTORY_REG(Median)
}  // namespace ge
#endif