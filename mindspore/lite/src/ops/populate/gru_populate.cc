/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/gru_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateGruParameter(const void *prim) {
  GruParameter *gru_param = reinterpret_cast<GruParameter *>(malloc(sizeof(GruParameter)));
  if (gru_param == nullptr) {
    MS_LOG(ERROR) << "malloc GruParameter failed.";
    return nullptr;
  }
  memset(gru_param, 0, sizeof(GruParameter));
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  gru_param->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_GRU();
  if (param == nullptr) {
    free(gru_param);
    MS_LOG(ERROR) << "get Gru param nullptr.";
    return nullptr;
  }
  gru_param->bidirectional_ = param->bidirectional();
  return reinterpret_cast<OpParameter *>(gru_param);
}
}  // namespace

Registry g_gruParameterRegistry(schema::PrimitiveType_GRU, PopulateGruParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
