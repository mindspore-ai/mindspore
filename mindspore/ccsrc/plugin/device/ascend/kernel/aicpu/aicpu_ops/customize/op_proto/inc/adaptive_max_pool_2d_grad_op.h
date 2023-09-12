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

#ifndef CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_MAX_POOL_2D_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_MAX_POOL_2D_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs the backpropagation of AdaptiveMaxPool2dGrad. \n

* @par Inputs:
* @li input_grad: A 3D or 4D Tensor of input's gradient. Must be one of RealNumberType. \n
* @li x: A 3D or 4D Tensor. Must be one of RealNumberType. \n
* @li argmax: A 3D or 4D Tensor of type IndexNumberType. \n

* @par Outputs:
* @li output_grad: A Tensor. Has the same data type and shape as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool2dGrad.
*/
REG_CUST_OP(AdaptiveMaxPool2dGrad)
  .INPUT(y_grad, TensorType::FloatingDataType())
  .INPUT(x, TensorType::FloatingDataType())
  .INPUT(argmax, TensorType::IndexNumberType())
  .OUTPUT(x_grad, TensorType::FloatingDataType())
  .CUST_OP_END_FACTORY_REG(AdaptiveMaxPool2dGrad)
}  // namespace ge
#endif