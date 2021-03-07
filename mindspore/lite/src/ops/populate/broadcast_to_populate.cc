/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/broadcast_to_fp32.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateBroadcastToParameter(const void *prim) {
  BroadcastToParameter *broadcast_param =
    reinterpret_cast<BroadcastToParameter *>(malloc(sizeof(BroadcastToParameter)));
  if (broadcast_param == nullptr) {
    MS_LOG(ERROR) << "malloc BroadcastToParameter failed.";
    return nullptr;
  }
  memset(broadcast_param, 0, sizeof(BroadcastToParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_BroadcastTo();
  broadcast_param->op_parameter_.type_ = primitive->value_type();
  auto dst_shape = value->shape();
  broadcast_param->shape_size_ = dst_shape->size();
  for (size_t i = 0; i < broadcast_param->shape_size_; ++i) {
    broadcast_param->shape_[i] = dst_shape->Get(i);
  }
  return reinterpret_cast<OpParameter *>(broadcast_param);
}

Registry BroadcastToParameterRegistry(schema::PrimitiveType_BroadcastTo, PopulateBroadcastToParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
