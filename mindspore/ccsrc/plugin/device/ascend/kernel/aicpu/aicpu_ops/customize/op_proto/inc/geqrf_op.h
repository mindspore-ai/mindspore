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

#ifndef CUSTOMIZE_OP_PROTO_INC_GEQRF_OP_H
#define CUSTOMIZE_OP_PROTO_INC_GEQRF_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Computes the Geqrf decompositions of one matrix.

* @par Inputs:
* @li x: A Tensor whose shape is [M, N].The inpute shape of x must be 2D.

* @par Outputs:
* @li r: A Tensor. Has the same type as x.
* @li tau: A Vector. Has min(M, N) elements.

* @par Third-party framework compatibility
* Compatible with pytorch Geqrf operator.
*/

REG_CUST_OP(Geqrf)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(tau, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(Geqrf)
}  // namespace ge
#endif