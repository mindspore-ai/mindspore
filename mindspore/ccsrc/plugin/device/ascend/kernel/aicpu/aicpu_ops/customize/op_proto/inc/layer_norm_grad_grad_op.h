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

#ifndef CUSTOMIZE_OP_PROTO_INC_LAYER_NORM_GRAD_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_LAYER_NORM_GRAD_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
 * @brief Compute the backward of LayerNormGrad. \n
 * @par Inputs:
 * x: the input x from LayerNorm. Must be one of the following types: float16, float32. \n
 * dy: the gradient of y. Must be one of the following types: float16,float32. \n
 * variance: the variance of x. Must be one of the following types: float16, float32. \n
 * mean: the mean value of x. Must be one of the following types: float16, float32. \n
 * gamma: the input gamma from LayerNorm. Must be one of the following types: float16, float32. \n
 * d_dx: the gradient of dx. Must be one of the following types: float16, float32. \n
 * d_dg: the gradient of dg. Must be one of the following types: float16, float32. \n
 * d_db: the gradient of db. Must be one of the following types: float16, float32. \n
 *
 * @par Outputs:
 * sopd_x: the gradient of x. Must be one of the following types: float16, float32. \n
 * sopd_dy: the gradient of dy. Must be one of the following types: float16, float32. \n
 * sopd_gamma: the gradient of gamma. Must be one of the following types: float16, float32. \n
 */
REG_CUST_OP(LayerNormGradGrad)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(d_dx, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(d_dg, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(d_db, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(sopd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(sopd_dy, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(sopd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(LayerNormGradGrad)
}  // namespace ge
#endif