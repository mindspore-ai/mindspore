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

#ifndef CUSTOMIZE_OP_PROTO_INC_TRACE_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_TRACE_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief  Computes the grad of x1 in trace. \n

* @par Inputs:
* Four inputs, including:
* @li y_grad: A tensor. \n
* @li x_shape: A tensor. Must be one of the following types:
*     int32, int64. \n

* @par Outputs:
* x_grad: A Tensor with the same type and shape of y_grad's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Trace Backward. \n
*/
REG_CUST_OP(TraceGrad)
  .INPUT(y_grad, TensorType::BasicType())
  .INPUT(x_shape, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(x_grad, TensorType::BasicType())
  .CUST_OP_END_FACTORY_REG(TraceGrad)
}  // namespace ge
#endif