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
* @brief Creates a one-dimensional tensor of size steps whose values are
    evenly spaced from start to end, inclusive, on a logarithmic scale
    with base base.

* @par Inputs:
* Two inputs, including:
* start: A tensor. Must be one of the following types:
*     float16, float32.  \n
* end: A tensor. Must be one of the following types:
*     float16, float32.  \n

* @par Attributes:
* @li steps: An optional int.Defaults to 100. \n
* @li base: An optional float.Defaults to 10.0. \n
* @li dtype: An optional int.Defaults to 1. \n

* @par Outputs:
* y: A Tensor with the same type and shape of input_x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator logspace. \n
*/
REG_CUST_OP(LogSpace)
  .INPUT(start, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(end, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .ATTR(steps, Int, 100)
  .ATTR(base, Int, 10)
  .ATTR(dtype, Int, 1)
  .CUST_OP_END_FACTORY_REG(LogSpace)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H