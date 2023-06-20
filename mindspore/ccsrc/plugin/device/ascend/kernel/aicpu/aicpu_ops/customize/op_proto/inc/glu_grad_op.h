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

#ifndef CUSTOMIZE_OP_PROTO_INC_GLU_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_GLU_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Counts the number of occurrences of each value in an integer array. \n

* @par Inputs:
* @li grads: A Tensor of grad_output. Its data tpye must be float16, float, double.
* @li x: A Tensor of input. Its data tpye must be float16, float, double.

* @par Outputs:
* @li output: A Tensor of grad_input with the same shape of x. Its data tpye must be float16, float, double.

* @par Attributes:
* binary_output: An required value for computing.


* @par Third-party framework compatibility
* the pytorch framework does not have the same operation.
*/
REG_CUST_OP(GluGrad)
  .INPUT(grads, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(axis, Int)
  .CUST_OP_END_FACTORY_REG(GluGrad)
}  // namespace ge
#endif