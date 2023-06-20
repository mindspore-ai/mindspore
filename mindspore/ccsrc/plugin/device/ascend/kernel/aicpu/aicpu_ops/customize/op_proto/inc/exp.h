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
* @brief Computes the exponential of "x" element-wise. \n

* @par Inputs:
* One input:\n
* x: A Tensor. Must be one of the following types: float16, float32, double, complex64, complex128. \n

* @par Attributes:
* @li base: An optional attribute of type float32, specifying the base gamma. Defaults to "-1.0".
* @li scale: An optional attribute of type float32, specifying the scale alpha. Defaults to "1.0".
* @li shift: An optional attribute of type float32, specifying the shift beta. Defaults to "0.0". \n

* @par Outputs:
* y: A Tensor of the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Exp.
*/
REG_CUST_OP(Exp)
  .INPUT(x, TensorType::UnaryDataType())
  .OUTPUT(y, TensorType::UnaryDataType())
  .ATTR(base, Float, -1.0)
  .ATTR(scale, Float, 1.0)
  .ATTR(shift, Float, 0.0)
  .CUST_OP_END_FACTORY_REG(Exp)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H