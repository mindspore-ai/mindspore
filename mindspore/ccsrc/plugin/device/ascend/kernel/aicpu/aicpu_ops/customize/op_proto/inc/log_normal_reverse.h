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
* @brief Draws samples from a multinomial distribution. \n
* @brief Fills the elements of the input tensor with log normal values initialized by given mean and std \n

* @par Inputs:
* Four inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float64 \n
* @li dim: A Tensor of type float, which is the mean of normal distribution. \n
* @li std: A Tensor of type float, which is the mean of normal distribution. \n
* @par Outputs:
* y: A Tensor with the same type and shape of input_x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator LogNormal. \n
*/
REG_CUST_OP(LogNormalReverse)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .ATTR(mean, Float, 2.0)
  .ATTR(std, Float, 1.0)
  .CUST_OP_END_FACTORY_REG(LogNormalReverse)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H