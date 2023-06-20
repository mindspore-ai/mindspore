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

#ifndef CUSTOMIZE_OP_PROTO_INC_LOGICAL_XOR_OP_H
#define CUSTOMIZE_OP_PROTO_INC_LOGICAL_XOR_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the element-wise LogicalXOR of the given input tensors.
Zeros are treated as False and nonzeros are treated as True. \n

*
* @par Inputs:
* @li x1: A tensor of type bool.
* @li x2: A tensor of the same type as "x1".
*
* @attention Constraints:
* LogicalXor supports broadcasting.
*
* @par Outputs:
* y: A tensor of the same type as "x1".
*
* @par Third-party framework compatibility
* Compatible with the Pytorch operator LogicalXor.
*
*/
REG_CUST_OP(LogicalXor)
  .INPUT(x1, TensorType({DT_BOOL}))
  .INPUT(x2, TensorType({DT_BOOL}))
  .OUTPUT(y, TensorType({DT_BOOL}))
  .CUST_OP_END_FACTORY_REG(LogicalXor)
}  // namespace ge
#endif