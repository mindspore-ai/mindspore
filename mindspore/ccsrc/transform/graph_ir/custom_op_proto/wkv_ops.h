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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_WKV_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_WKV_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

/* clang-format off */

namespace ge {
REG_OP(Wkv)
  .INPUT(w, ge::TensorType::ALL())
  .INPUT(u, ge::TensorType::ALL())
  .INPUT(k, ge::TensorType::ALL())
  .INPUT(v, ge::TensorType::ALL())
  .INPUT(m, ge::TensorType::ALL())
  .INPUT(p, ge::TensorType::ALL())
  .INPUT(q, ge::TensorType::ALL())
  .OUTPUT(o, ge::TensorType::ALL())
  .OUTPUT(mo, ge::TensorType::ALL())
  .OUTPUT(po, ge::TensorType::ALL())
  .OUTPUT(qo, ge::TensorType::ALL())
  .OP_END_FACTORY_REG(Wkv);

REG_OP(WkvGrad)
  .INPUT(w, ge::TensorType::ALL())
  .INPUT(u, ge::TensorType::ALL())
  .INPUT(k, ge::TensorType::ALL())
  .INPUT(v, ge::TensorType::ALL())
  .INPUT(gy, ge::TensorType::ALL())
  .OUTPUT(gw, ge::TensorType::ALL())
  .OUTPUT(gu, ge::TensorType::ALL())
  .OUTPUT(gk, ge::TensorType::ALL())
  .OUTPUT(gv, ge::TensorType::ALL())
  .OP_END_FACTORY_REG(WkvGrad);
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_WKV_OPS_H_
