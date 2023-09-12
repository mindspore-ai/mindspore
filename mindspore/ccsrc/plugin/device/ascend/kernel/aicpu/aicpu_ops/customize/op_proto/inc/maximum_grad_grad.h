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
* @brief Calculates the reversed outputs of the function "MaximumGradGrad". \n

* @par Inputs:
* Four inputs, including:
* @li grad_y1: A mutable Tensor. Must be one of the following types:
*     float16, float32, int32.
* @li grad_y2: A mutable Tensor. Has the same type as "grad_y1".
* @li x1: A mutable Tensor of the same type as "grad_y1".
* @li x2: A mutable Tensor of the same type as "grad_y1". \n

* @par Outputs:
* @li spod_x1: A mutable Tensor. Has the same type as "grad_y1".
* @li spod_x2: A mutable Tensor. Has the same type as "grad_y1".
* @li sopd_grads: A mutable Tensor. Has the same type as "grad_y1". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaximumGradGrad.
*/
REG_CUST_OP(MaximumGradGrad)
  .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .INPUT(grad_y1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .INPUT(grad_y2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(spod_x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(spod_x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(spod_grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .CUST_OP_END_FACTORY_REG(MaximumGradGrad)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H