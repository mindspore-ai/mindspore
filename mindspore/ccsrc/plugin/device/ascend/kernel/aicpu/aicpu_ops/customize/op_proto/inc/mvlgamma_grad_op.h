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

#ifndef CUSTOMIZE_OP_PROTO_INC_MVLGAMMA_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MVLGAMMA_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes gradients for Mvlgamma (multivariate log-gamma function). \n

* @par Inputs:
* y_grad:A Tensor. Must be one of the following types: float32, double.
* x:A Tensor. Must be one of the following types: float32, double.

* @par Attributes:
* p:A required attribute of the following types: int32, int64. \n

* @par Outputs:
* x_grad:A Tensor. Has the same type as y_grad and x. \n

* @par Third-party framework compatibility.
* Compatible with pytorch mvlgamma_backward operator.
*/

REG_CUST_OP(MvlgammaGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT, DT_DOUBLE}))
  .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_DOUBLE}))
  .REQUIRED_ATTR(p, Int)
  .CUST_OP_END_FACTORY_REG(MvlgammaGrad)
}  // namespace ge
#endif