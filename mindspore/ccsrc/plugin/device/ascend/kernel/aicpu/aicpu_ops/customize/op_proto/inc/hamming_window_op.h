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

#ifndef CUSTOMIZE_OP_PROTO_INC_HAMMING_WINDOW_OP_H
#define CUSTOMIZE_OP_PROTO_INC_HAMMING_WINDOW_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the hamming_window function. \n

* @par Inputs:
* length: A Tensor of IntegerDataType, the size of returned window. \n

* @par Attributes:
* @li periodic: An optional flag, if True, returns a window to be used as periodic
*     function. If False, return a symmetric window. Defaults to True. \n
* @li alpha: An optional float coefficient. Defaults to 0.54. \n
* @li beta: An optional float coefficient. Defaults to 0.46. \n
* @li dtype: The desired data type of returned tensor. Only floating point
*     types are supported. Defaults to "float". \n

* @par Outputs:
* y: A Tensor with type as attribute dtype. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator hamming_window. \n
*/
REG_CUST_OP(HammingWindow)
  .INPUT(length, TensorType::IntegerDataType())
  .OUTPUT(y, TensorType::FloatingDataType())
  .ATTR(periodic, Bool, true)
  .ATTR(alpha, Float, 0.54)
  .ATTR(beta, Float, 0.46)
  .ATTR(dtype, Int, 0)
  .CUST_OP_END_FACTORY_REG(HammingWindow)
}  // namespace ge
#endif