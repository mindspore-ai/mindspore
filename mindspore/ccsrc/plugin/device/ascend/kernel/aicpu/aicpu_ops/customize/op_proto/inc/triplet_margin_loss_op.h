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

#ifndef CUSTOMIZE_OP_PROTO_INC_TRIPLET_MARGIN_LOSS_OP_H
#define CUSTOMIZE_OP_PROTO_INC_TRIPLET_MARGIN_LOSS_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes Triplet Margin loss.
* @par Inputs:
* three inputs
* @li x: An ND Tensor of basic type.(x1,x2,...,xn),n should be less than 9.
* @li positive: An ND Tensor of basic type.(x1,x2,...,xn),n should be less than 9.
* @li negative: An ND Tensor of basic type.(x1,x2,...,xn),n should be less than 9.
* @li margin :A tensor of float32,shape:{1}.

* @par Attributes:
* p :An optional int .The norm degree for pairwise distance. Defaults to 2.
* swap :An optional bool. The distance swap is described in detail in the
* paper Learning shallow convolutional feature descriptors with triplet
* losses by V. Balntas, E. Riba et al. Defaults to false.
* reduction: A character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean". \n
* eps :An optional float ,Defaults to 1e-6. \n

* @par Outputs:
* y: When reduction="sum" or "mean", y is a scalar.
* When reduction = "none", shape of y is (x1,x3,....,xn). \n
*/
REG_CUST_OP(TripletMarginLoss)
  .INPUT(x, TensorType::BasicType())
  .INPUT(positive, TensorType::BasicType())
  .INPUT(negative, TensorType::BasicType())
  .INPUT(margin, TensorType({DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .ATTR(p, Int, 2)
  .ATTR(swap, Bool, false)
  .ATTR(eps, Float, 1e-6)
  .ATTR(reduction, String, "mean")
  .CUST_OP_END_FACTORY_REG(TripletMarginLoss)
}  // namespace ge
#endif