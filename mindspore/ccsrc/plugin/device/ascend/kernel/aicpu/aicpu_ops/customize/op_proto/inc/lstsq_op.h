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

#ifndef CUSTOMIZE_OP_PROTO_INC_LSTSQ_OP_H
#define CUSTOMIZE_OP_PROTO_INC_LSTSQ_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @Calculates the solutions of the least squares and minimum norm problems. \n

* @par Inputs:
* matrix: An 2D tensor of type float16, float32, double.
* rhs: An 2D tensor of type float16, float32, double.

* @par Attributes:
* @li Ie_regularizer: An optional float. This value defaults to 0.0.
* @li fast: An optional bool. This value defaults to True.

* @par Outputs:
* y: An 2D tensor of type float16, float32, double.
*/

REG_CUST_OP(Lstsq)
  .INPUT(matrix, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(rhs, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .ATTR(l2_regularizer, Float, 0.0)
  .ATTR(fast, Bool, true)
  .CUST_OP_END_FACTORY_REG(Lstsq)
}  // namespace ge
#endif