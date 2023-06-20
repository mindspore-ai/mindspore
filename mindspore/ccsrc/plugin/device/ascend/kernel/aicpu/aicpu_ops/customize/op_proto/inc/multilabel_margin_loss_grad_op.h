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

#ifndef CUSTOMIZE_OP_PROTO_INC_MULTILABEL_MARGIN_LOSS_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MULTILABEL_MARGIN_LOSS_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the MultilabelMarginLossGrad . \n

* @par Inputs:
* @li y_grad: A scalar of type float16, float32. \n
* @li x: A 1D or 2D Tensor of dtype float16, float32. \n
* @li target: A 1D or 2D Tensor of dtype int32. \n
* @li is_target: A 1D or 2D Tensor of dtype int32. \n

* @par Attributes:
* reduction:A character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean". \n

* @par Outputs:
* x_grad: A Tensor. The same type and shape as "x". \n
*/
REG_CUST_OP(MultilabelMarginLossGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(target, TensorType({DT_INT32}))
  .INPUT(is_target, TensorType({DT_INT32}))
  .ATTR(reduction, String, "mean")
  .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(MultilabelMarginLossGrad)
}  // namespace ge
#endif