/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/ragged_range_fp32.h"
using mindspore::schema::PrimitiveType_RaggedRange;

namespace mindspore {
namespace lite {
OpParameter *PopulateRaggedRangeParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto *param = reinterpret_cast<RaggedRangeParameter *>(malloc(sizeof(RaggedRangeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc RaggedRangeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(RaggedRangeParameter));

  param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_RaggedRange, PopulateRaggedRangeParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
