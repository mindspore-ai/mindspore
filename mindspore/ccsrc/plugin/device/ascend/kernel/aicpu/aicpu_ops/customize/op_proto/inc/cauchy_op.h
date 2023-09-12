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

#ifndef CUSTOMIZE_OP_PROTO_INC_CAUCHY_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CAUCHY_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the output as cauchy distribution . \n

* @par Attributes:
* @li sigma: Optional. Must be one of the following types: float32. Defaults to 1.0.
* @li median: Optional. Must be one of the following types: float32. Defaults to 0.0.
* @li size: Required. Must be one of the following types: listint . \n

* @par Outputs:
* @li y: A Tensor. types:float16, float32.
*/
REG_CUST_OP(Cauchy)
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .ATTR(sigma, Float, 1.0)
  .ATTR(median, Float, 0.0)
  .REQUIRED_ATTR(size, ListInt)
  .CUST_OP_END_FACTORY_REG(Cauchy);
}  // namespace ge
#endif