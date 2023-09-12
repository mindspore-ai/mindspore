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

#ifndef CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_MAX_POOL3_D_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_MAX_POOL3_D_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs the backpropagation of AdaptiveMaxPool3dGrad. \n

* @par Inputs:
* @li input_grad: A 4D or 5D Tensor of input's gradient. Must be one of RealNumberType. \n
* @li x: A 4D or 5D Tensor. Must be one of RealNumberType. \n
* @li argmax: A 4D or 5D Tensor of type int32. \n

* @par Outputs:
* @li output_grad: A Tensor. Has the same data type and shape as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool3dGrad.
*/
REG_CUST_OP(AdaptiveMaxPool3dGrad)
  .INPUT(input_grad, TensorType::RealNumberType())
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(argmax, TensorType({DT_INT32}))
  .OUTPUT(output_grad, TensorType::RealNumberType())
  .CUST_OP_END_FACTORY_REG(AdaptiveMaxPool3dGrad)
}  // namespace ge
#endif