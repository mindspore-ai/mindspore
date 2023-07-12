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

#ifndef CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_MAX_POOL3D_OP_H
#define CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_MAX_POOL3D_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Applies a 3D adaptive max pooling over an input signal composed of several input planes. \n
* The output is of size D x H x W, for 4D/5D input Tensor. \n
* The number of output features is equal to the number of input planes.

* @par Inputs:
* Two inputs, including:
* @li x: A 4D/5D Tensor. Must be one of RealNumberType.
* @li output_size: A 3D Tensor with data type int32.

* @par Outputs:
* Two outputs, including:
* @li y: A Tensor. Has the same data type as "x" \n
* @li argmax: A Tensor with data type int32. Has the same shape as "y"

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool3d.
*/
REG_CUST_OP(AdaptiveMaxPool3d)
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(output_size, TensorType({DT_INT32}))
  .OUTPUT(y, TensorType::RealNumberType())
  .OUTPUT(argmax, TensorType({DT_INT32}))
  .CUST_OP_END_FACTORY_REG(AdaptiveMaxPool3d)
}  // namespace ge
#endif