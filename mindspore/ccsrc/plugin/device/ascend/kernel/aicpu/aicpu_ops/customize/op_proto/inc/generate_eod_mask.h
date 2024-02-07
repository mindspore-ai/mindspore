/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_GENERATE_EOD_MASK_H
#define CUSTOMIZE_OP_PROTO_INC_GENERATE_EOD_MASK_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief flip the bit. \n

* @par Inputs:
* One input, including:
* @li x:A Tensor. Must be one of the following types: float32, float64, bfloat16.
 Shape is 2D tensor. \n

* @par Attributes:
* @li compute_v: A bool. Indicating whether to compute eigenvectors. \n

* @par Outputs:
* output: A Tensor. The modified tensor. \n
*/

REG_CUST_OP(GenerateEodMask)
  .INPUT(inputs_ids, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(position_ids,
          TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
  .ATTR(n_pos, Int, false)
  .ATTR(eod_token_id, Int, false)
  .ATTR(n_step, ListInt, {})
  .ATTR(n_error_mode, String, "specific")
  .CUST_OP_END_FACTORY_REG(GenerateEodMask)
}  // namespace ge
#endif
