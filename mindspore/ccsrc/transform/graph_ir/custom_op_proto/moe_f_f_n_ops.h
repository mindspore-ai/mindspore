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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_MOE_F_F_N_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_MOE_F_F_N_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

/* clang-format off */

namespace ge {
REG_OP(MoeFFN)
  .INPUT(x, TensorType({DT_INT8, DT_FLOAT16}))
  .INPUT(expert_tokens, TensorType({DT_INT64}))
  .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16}))
  .OPTIONAL_INPUT(bias1, TensorType({DT_FLOAT16}))
  .OPTIONAL_INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16}))
  .OPTIONAL_INPUT(bias2, TensorType({DT_FLOAT16}))
  .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16}))
  .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16}))
  .OPTIONAL_INPUT(deq_scale1, TensorType({DT_FLOAT16}))
  .OPTIONAL_INPUT(deq_scale2, TensorType({DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT16}))
  .REQUIRED_ATTR(activation, String)
  .OP_END_FACTORY_REG(MoeFFN);
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_MOE_F_F_N_OPS_H_
