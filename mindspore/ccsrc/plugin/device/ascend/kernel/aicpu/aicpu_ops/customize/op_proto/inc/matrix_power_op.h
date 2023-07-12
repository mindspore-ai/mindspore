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

#ifndef CUSTOMIZE_OP_PROTO_INC_MATRIX_POWER_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MATRIX_POWER_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief: Computes the n-th power of a batch of square matrices. \n

* @par Inputs:
* @li x: A tensor of shape (n, m, m). Must be one of the following types: float32, float16. \n

* @par Outputs:
* @li y: A tensor of the same shape and type with x. \n

* @par Attributes:
* @li n: A required int. The exponent. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator MatrixPower.
*/

REG_CUST_OP(MatrixPower)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(n, Int)
  .CUST_OP_END_FACTORY_REG(MatrixPower)
}  // namespace ge
#endif