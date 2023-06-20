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

#ifndef CUSTOMIZE_OP_PROTO_INC_FRACTIONAL_MAX_POOL_3D_GRAD_WITH_FIXED_KSIZE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_FRACTIONAL_MAX_POOL_3D_GRAD_WITH_FIXED_KSIZE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Compute gradients of FractionalMaxPool3d function. \n

* @par Inputs:
* @li origin_input: A Tensor. Must be one of the following types: int32, int64, supported format list ["NCDHW, NDHWC"].
* @li out_backprop: A Tensor. Must be one of the following types: float16,
* float32, double, int32, int64, supported format list ["NCDHW, NDHWC"].
* @li argmax: A Tensor. Must be one of the following types: int32, int64. \n

* @par Attributes:
* @li data_format: An optional string. Defaults to "NCDHW". \n

* @par Outputs:
* @li y: A Tensor. Has the same type as x.

* @par Third-party framework compatibility
* @li compatible with Pytorch FractionalMaxPool3dBackward operator.
*/
REG_CUST_OP(FractionalMaxPool3DGradWithFixedKsize)
  .INPUT(origin_input, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_INT32, DT_INT64}))
  .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_INT32, DT_INT64}))
  .INPUT(argmax, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_INT32, DT_INT64}))
  .ATTR(data_format, String, "NCDHW")
  .CUST_OP_END_FACTORY_REG(FractionalMaxPool3DGradWithFixedKsize);
}  // namespace ge
#endif