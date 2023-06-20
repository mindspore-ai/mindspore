/**
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_FRACTIONAL_MAX_POOL__3D_WITH_FIXED_KSIZE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_FRACTIONAL_MAX_POOL__3D_WITH_FIXED_KSIZE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs fractional max pooling on the input. \n

* @par Inputs:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
* int64, supported format list ["NCDHW, NDHWC"].
* @li random_samples: A Tensor. Must be one of the following types: float16, float32, double. \n

* @par Attributes:
* @li ksize: A required list of 3 floats. specifying the window size (D,H,W) of the input tensor.
* @li output_shape: A required list of 3 ints. specifying the size (D,H,W) of the output tensor.
* @li data_format: An optional string. Defaults to "NCDHW". \n

* @par Outputs:
* @li y: A Tensor. Has the same type as x.
* @li argmax: A Tensor of type int64 or int32. \n

* @par Third-party framework compatibility
* @li compatible with Pytorch FractionalMaxPool3d operator.
*/
REG_CUST_OP(FractionalMaxPool3DWithFixedKsize)
  .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_INT32, DT_INT64}))
  .INPUT(random_samples, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_INT32, DT_INT64}))
  .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
  .REQUIRED_ATTR(ksize, ListFloat)
  .REQUIRED_ATTR(output_shape, ListInt)
  .ATTR(data_format, String, "NCDHW")
  .CUST_OP_END_FACTORY_REG(FractionalMaxPool3DWithFixedKsize);
}  // namespace ge
#endif