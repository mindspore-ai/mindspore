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

#ifndef CUSTOMIZE_OP_PROTO_INC_BARTLETT_WINDOW_OP_H
#define CUSTOMIZE_OP_PROTO_INC_BARTLETT_WINDOW_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes Bartlett window function. \n

*
* @par Inputs:
* @li window_length: A tensor of IntegerDataType.
*
* @par Attributes:
* @li periodic: An optional attribute. Defaults to true .
* @li dtype: An optional attribute. Defaults to "0" .
*
*
* @par Outputs:
* y: A 1-D tensor of size (window_length,) containing the window
*
* @par Third-party framework compatibility
* Compatible with the Pytorch operator LogicalXor.
*
*/
REG_CUST_OP(BartlettWindow)
  .INPUT(window_length, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .ATTR(periodic, Bool, true)
  .ATTR(dtype, Int, 0)
  .CUST_OP_END_FACTORY_REG(BartlettWindow)
}  // namespace ge
#endif