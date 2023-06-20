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

#ifndef CUSTOMIZE_OP_PROTO_INC_MEDIAN_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MEDIAN_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief MedianGrad. \n

* @par Inputs:
* four inputs, including:
* @li y_grad: A Tensor. Support int16/int32/int64/float/double.\n
* @li x: A Tensor. Support int16/int32/int64/float/double.\n
* @li y: A Tensor. Support int16/int32/int64/float/double.\n
* @li indices: A Tensor. Support int32/int64.\n

* @par Outputs:
* one output, including:
* @li x_grad: A Tensor. Support int16/int32/int64/float/double.\n

* @par Attributes:
* @li global_median: whether the output is the global median of all elements or just in the dim.\n
* @li axis:the dimension to reduce.\n
* @li keepdim: whether the output tensor has dim retained or not.\n
*/
REG_CUST_OP(MedianGrad)
  .INPUT(y_grad, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
  .INPUT(x, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
  .INPUT(y, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
  .OPTIONAL_INPUT(indices, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(x_grad, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
  .REQUIRED_ATTR(global_median, Bool)
  .ATTR(axis, Int, 0)
  .ATTR(keepdim, Bool, false)
  .CUST_OP_END_FACTORY_REG(MedianGrad)
}  // namespace ge
#endif