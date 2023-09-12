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

#ifndef CUSTOMIZE_OP_PROTO_INC_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs fractional max pooling with fixed ksize on the input . \n

* @par Inputs:
* Inputs include:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32, int64.
* 4-D with shape [batch, channels, height, width] . \n
* @li random_samples: A Tensor. The value must > 0 and < 1. 3-D with shape [batch, channels, 2]

* @par Outputs:
* @li y: A Tensor. Has the same type as x.
* @li argmax: A Tensor. Has the same shape as y. Specifying the index of maximum value in input x
* Each element in y is a maximum value.

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for each dimension of the input tensor.
* @li output_shape: A required tuple or list, specifying the size of the output y.
* @li data_format: The default is "NCHW". Specifying the format for input x.
*/
REG_CUST_OP(FractionalMaxPoolWithFixedKsize)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
  .INPUT(random_samples, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
  .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
  .REQUIRED_ATTR(ksize, ListInt)
  .REQUIRED_ATTR(output_shape, ListInt)
  .ATTR(data_format, String, "NCHW")
  .CUST_OP_END_FACTORY_REG(FractionalMaxPoolWithFixedKsize)
}  // namespace ge
#endif