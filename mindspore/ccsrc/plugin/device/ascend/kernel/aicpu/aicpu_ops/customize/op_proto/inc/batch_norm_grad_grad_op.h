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

#ifndef CUSTOMIZE_OP_PROTO_INC_BATCH_NORM_GRAD_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_BATCH_NORM_GRAD_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs the backpropagation of BatchNormGrad. \n

* @par Notes:
* @li Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
*     The size of 1D Tensors matches the dimension C of the 4D Tensors. \n

* @par Inputs:
* @li x: the input "x" from BatchNormGrad. Must be a 4D tensor.
* @li dy: the input "y_backprop" from BatchNormGrad. Must be a 4D tensor.
* @li scale: the input "scale" from BatchNormGrad. Must be a 1D tensor.
* @li reserve_space_1: If "is_training" is true, input the batch-mean of "x".
*     If "is_training" is false, input the running-mean of "x". Must be a 1D tensor.
* @li reserve_space_2: If "is_training" is true, input the batch-var of "x".
*     If "is_training" is false, input the running-var of "x". Must be a 1D tensor.
* @li ddx: the output "x_backprop" from BatchNormGrad. Must be a 4D tensor.
* @li ddscale: the output "scale_backprop" from BatchNormGrad. Must be a 1D tensor.
* @li ddoffset: the output "offset_backprop" from BatchNormGrad. Must be a 1D tensor. \n

* @par Attributes:
* @li epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".
* @li data_format: An optional string, ranging from "NHWC" "NCHW". Defaults to "NHWC".
* @li is_training: An optional bool. Defaults to "true". Specifies the operation is for training or inference . \n

* @par Outputs:
* @li dx: the gradient of x. It has the same tensor desc as "x".
* @li ddy: the gradient of dy. It has the same tensor desc as "dy".
* @li dscale: the gradient of scale. It has the same tensor desc as "scale". \n
*/
REG_CUST_OP(BatchNormGradGrad)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(scale, TensorType({DT_FLOAT}))
  .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
  .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
  .INPUT(ddx, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(ddscale, TensorType({DT_FLOAT}))
  .INPUT(ddoffset, TensorType({DT_FLOAT}))
  .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(ddy, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(dscale, TensorType({DT_FLOAT}))
  .ATTR(epsilon, Float, 0.0001)
  .ATTR(data_format, String, "NHWC")
  .ATTR(is_training, Bool, true)
  .CUST_OP_END_FACTORY_REG(BatchNormGradGrad)
}  // namespace ge
#endif