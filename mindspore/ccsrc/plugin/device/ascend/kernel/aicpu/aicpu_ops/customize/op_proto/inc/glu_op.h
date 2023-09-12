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

#ifndef CUSTOMIZE_OP_PROTO_INC_GLU_OP_H
#define CUSTOMIZE_OP_PROTO_INC_GLU_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
   * @brief The Glu operator represents a gated linear unit. Where the input is divided into two halves to
   * form A and B, All elements in B are evaluated by the sigmoid function, A and B do elements-wise product . \n

   * @par Inputs:
   * @li x: A tensor of type float16, float32, float64 . \n

   * @par Attributes:
   * axis: An optional attribute int32. Specifies the dimension along which to split. Defaults to -1 . \n

   * @par Outputs:
   * @li y:  output with the same dtype of input x. \n

   * @attention Constraints:
   * @li "axis" is in the range [-len(x.shape), (x.shape)-1] . \n

   * @par Third-party framework compatibility
   * Compatible with the PyTorch operator GLU.
   */

REG_CUST_OP(Glu)
  .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16}))
  .ATTR(axis, Int, -1)
  .CUST_OP_END_FACTORY_REG(Glu)
}  // namespace ge
#endif