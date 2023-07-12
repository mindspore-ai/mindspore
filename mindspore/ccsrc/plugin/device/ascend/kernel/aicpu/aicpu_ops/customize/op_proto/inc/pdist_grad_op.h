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

#ifndef CUSTOMIZE_OP_PROTO_INC_PDIST_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_PDIST_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the grad of x in pdist. \n

* @par Inputs:
* Thress inputs, including:
* @li grad: A tensor. Must be one of the following types:
*     float16, float32. \n
* @li x: A tensor. Must be one of the following types:
*     float16, float32. \n
* @li pdist: Output tensor of cdist forward.
*     Must be one of the following types: float16, float32. \n

* @par Attributes:
* p: An optional float.Defaults to 2. \n

* @par Outputs:
* y: A Tensor with the same type and shape of x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Pdist Backward. \n
*/
REG_CUST_OP(PdistGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(pdist, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .ATTR(p, Float, 2.0)
  .CUST_OP_END_FACTORY_REG(PdistGrad)
}  // namespace ge
#endif