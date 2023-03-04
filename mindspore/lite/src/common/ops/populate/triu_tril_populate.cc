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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/triu_tril.h"
using mindspore::schema::PrimitiveType_Tril;
using mindspore::schema::PrimitiveType_Triu;

namespace mindspore {
namespace lite {
OpParameter *PopulateTriuParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);

  auto *param = reinterpret_cast<TriuParameter *>(malloc(sizeof(TriuParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(TriuParameter));

  param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(param);
}
// cppcheck-suppress unknownMacro
REG_POPULATE(PrimitiveType_Triu, PopulateTriuParameter, SCHEMA_CUR)

OpParameter *PopulateTrilParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);

  auto *param = reinterpret_cast<TrilParameter *>(malloc(sizeof(TrilParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(TrilParameter));

  param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(param);
}
// cppcheck-suppress unknownMacro
REG_POPULATE(PrimitiveType_Tril, PopulateTrilParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
