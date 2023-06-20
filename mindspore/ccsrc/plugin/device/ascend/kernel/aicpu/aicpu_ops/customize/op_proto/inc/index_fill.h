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

#ifndef CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Fills the elements of the input tensor with value val by selecting the indices in the order given in index. \n

* @par Inputs:
* Four inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, float64, uint8, uint16, uint32,
*     uint64, int8, int16, int32, int64. \n
* @li dim: A Tensor of type int32, dimension along which to index. \n
* @li indices: A Tensor of the indices, type should be int32. \n
* @li value: A tensor. Must be one of the following types: float16, float32, float64, uint8, uint16, uint32,
*     uint64, int8, int16, int32, int64. \n

* @par Outputs:
* y: A Tensor with the same type and shape of input_x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator IndexFill. \n
*/
REG_CUST_OP(IndexFill)
  .INPUT(x, TensorType::BasicType())
  .INPUT(dim, TensorType({DT_INT32}))
  .INPUT(indices, TensorType({DT_INT32}))
  .INPUT(value, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .CUST_OP_END_FACTORY_REG(IndexFill)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H