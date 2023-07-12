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
* @brief InstanceNormV2Grad operator interface implementation.

* @par Inputs:
* Seven inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float32.
* @li mean: A Tensor. Must be one of the following types: float32.
* @li gamma: A Tensor. Must be one of the following types: float32.
* @li save_mean: A Tensor. Must be one of the following types: float32.
* @li save_variance: A Tensor. Must be one of the following types: float32.

* @par Attributes:
* @li is_training: An optional bool, specifying if the operation is used for
* training or inference. Defaults to "True".
* @li epsilon: An optional float32, specifying the small value added to
* variance to avoid dividing by zero. Defaults to "0.00001".

* @par Outputs:
* Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
* @li pd_gamma: A Tensor. Must be one of the following types: float32.
* @li pd_beta: A Tensor. Must be one of the following types: float32.

* @par Restrictions:
* Warning: THIS FUNCTION NOW IS SUPPORT 5D INPUT WITH FORMAT NC1HWC0 AND 4D INPUT WITH FORMAT NCHW & NHWC
*/
REG_CUST_OP(InstanceNormV2Grad)
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(gamma, TensorType({DT_FLOAT}))
  .INPUT(mean, TensorType({DT_FLOAT}))
  .INPUT(variance, TensorType({DT_FLOAT}))
  .INPUT(save_mean, TensorType({DT_FLOAT}))
  .INPUT(save_variance, TensorType({DT_FLOAT}))

  .OUTPUT(pd_x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(pd_gamma, TensorType({DT_FLOAT}))
  .OUTPUT(pd_beta, TensorType({DT_FLOAT}))

  .ATTR(is_training, Bool, true)
  .ATTR(epsilon, Float, 0.00001)
  .CUST_OP_END_FACTORY_REG(InstanceNormV2Grad)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H