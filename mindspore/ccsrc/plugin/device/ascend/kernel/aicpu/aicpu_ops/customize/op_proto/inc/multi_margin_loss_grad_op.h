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

#ifndef CUSTOMIZE_OP_PROTO_INC_MULTI_MARGIN_LOSS_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MULTI_MARGIN_LOSS_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the MultiMarginLossGrad. \n

* @par Inputs:
* @li y_grad: A scalar of type float16, float32, double.
* @li x: A 2D Tensor of dtype float16, float32, double.
* @li target: A 1D Tensor of dtype int64.
* @li weight:Optional 1D Tensor of dtype float16, float32, double.
a manual rescaling weight given to each class.
* If given, it has to be a Tensor of size C. Otherwise,
* it is treated as if having all ones. \n

* @par Attributes:
* margin :An optional float , Defaults to 1.
* p :An optional int .The norm degree for pairwise distance.Should be 1 or 2.
Defaults to 1.
* reduction:A character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean". \n


* @par Outputs:
* x_grad: A Tensor.  the same type and shape as "x".
*/

REG_CUST_OP(MultiMarginLossGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(target, TensorType({DT_INT64}))
  .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .ATTR(p, Int, 1)
  .ATTR(margin, Float, 1.0)
  .ATTR(reduction, String, "mean")
  .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(MultiMarginLossGrad)
}  // namespace ge
#endif