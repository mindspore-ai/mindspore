/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/attention_parameter.h"

using mindspore::schema::PrimitiveType_Attention;

namespace mindspore {
namespace lite {
OpParameter *PopulateAttentionParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  auto value = primitive->value_as_Attention();
  MS_CHECK_TRUE_MSG(value != nullptr, nullptr, "value is nullptr.");
  auto *param = reinterpret_cast<AttentionParameter *>(malloc(sizeof(AttentionParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc AttentionParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(AttentionParameter));
  param->op_parameter_.type_ = primitive->value_type();
  param->head_num_ = value->head_num();
  param->head_size_ = value->head_size();
  param->cross_ = value->cross();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Attention, PopulateAttentionParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
