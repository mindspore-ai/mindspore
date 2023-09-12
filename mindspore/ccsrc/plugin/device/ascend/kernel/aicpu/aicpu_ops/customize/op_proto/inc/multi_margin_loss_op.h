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

#ifndef CUSTOMIZE_OP_PROTO_INC_MULTI_MARGIN_LOSS_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MULTI_MARGIN_LOSS_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes Multi Margin loss.
* @par Inputs:
* two inputs,one optional input including:
* @li x: with shape :math:`(N, C)`. Data type only support float32 and float16,float64
* @li target: Ground truth labels, with shape :math:`(N,)`. Data type only support int64.
* @li weight:The rescaling weight to each class, with shape :math:`(C,)` and data type only
          support float32 and float16,float64. \n

* @par Attributes:
* margin :An optional float , Defaults to 1.
* p :An optional int .The norm degree for pairwise distance. Should be 1 or 2. Defaults to 1.
* reduction:A character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean". \n

* @par Outputs:
* y: when reduction="sum" or "mean", y is a scalar. when reduction="none", y has the
* same  shape as "target". \n
*/
REG_CUST_OP(MultiMarginLoss)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(target, TensorType({DT_INT64}))
  .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .ATTR(p, Int, 1)
  .ATTR(margin, Float, 1.0)
  .ATTR(reduction, String, "mean")
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(MultiMarginLoss)
}  // namespace ge
#endif