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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/elu_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateEluParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto elu_prim = primitive->value_as_Elu();
  EluParameter *elu_parameter = reinterpret_cast<EluParameter *>(malloc(sizeof(EluParameter)));
  if (elu_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EluParameter failed.";
    return nullptr;
  }
  memset(elu_parameter, 0, sizeof(EluParameter));
  elu_parameter->op_parameter_.type_ = schema::PrimitiveType_Elu;

  elu_parameter->alpha_ = elu_prim->alpha();
  return reinterpret_cast<OpParameter *>(elu_parameter);
}
}  // namespace

Registry g_eluV0ParameterRegistry(schema::v0::PrimitiveType_Elu, PopulateEluParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
