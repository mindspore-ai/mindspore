/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_MSDA_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_MSDA_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

/* clang-format off */

namespace ge {
REG_OP(MultiScaleDeformableAttentionV2Grad)
  .INPUT(value, ge::TensorType::ALL())
  .INPUT(spatial_shapes, ge::TensorType::ALL())
  .INPUT(level_start_index, ge::TensorType::ALL())
  .INPUT(sampling_loc, ge::TensorType::ALL())
  .INPUT(attn_weight, ge::TensorType::ALL())
  .INPUT(grad_output, ge::TensorType::ALL())
  .OUTPUT(grad_value, ge::TensorType::ALL())
  .OUTPUT(grad_sampling_loc, ge::TensorType::ALL())
  .OUTPUT(grad_attn_weight, ge::TensorType::ALL())
  .OP_END_FACTORY_REG(MultiScaleDeformableAttentionV2Grad);

REG_OP(MultiScaleDeformableAttnFunctionV2)
  .INPUT(value, ge::TensorType::ALL())
  .INPUT(value_spatial_shapes, ge::TensorType::ALL())
  .INPUT(value_level_start_index, ge::TensorType::ALL())
  .INPUT(sampling_locations, ge::TensorType::ALL())
  .INPUT(attention_weights, ge::TensorType::ALL())
  .OUTPUT(output, ge::TensorType::ALL())
  .OP_END_FACTORY_REG(MultiScaleDeformableAttnFunctionV2);
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_MSDA_OPS_H_
