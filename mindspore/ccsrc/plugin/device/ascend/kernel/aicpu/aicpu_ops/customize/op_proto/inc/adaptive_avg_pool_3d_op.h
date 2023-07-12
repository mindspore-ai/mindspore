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

#ifndef CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_AVG_POOL_3D_OP_H
#define CUSTOMIZE_OP_PROTO_INC_ADAPTIVE_AVG_POOL_3D_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs average pooling on the last 3 dimension . \n

* @par Inputs:
* x: A tensor of type float16, float32, double, int8, uint8, int16, int32, int64 . \n
* output_size: The shape of the last 3 dim of output. A tensor of type int32. \n

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as input "x" . \n
*/

REG_CUST_OP(AdaptiveAvgPool3d)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
  .INPUT(output_size, TensorType({DT_INT32}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(AdaptiveAvgPool3d)

    REG_CUST_OP(Constant)
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                         DT_UINT64, DT_BOOL, DT_DOUBLE}))
  .ATTR(value, Tensor, Tensor())
  .CUST_OP_END_FACTORY_REG(Constant)
}  // namespace ge
#endif