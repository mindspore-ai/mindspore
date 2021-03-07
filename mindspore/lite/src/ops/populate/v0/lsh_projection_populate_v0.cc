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
#include "nnacl/lsh_projection_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateLshProjectionParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto lsh_projection_prim = primitive->value_as_LshProjection();
  LshProjectionParameter *lsh_project_param =
    reinterpret_cast<LshProjectionParameter *>(malloc(sizeof(LshProjectionParameter)));
  if (lsh_project_param == nullptr) {
    MS_LOG(ERROR) << "malloc LshProjectionParameter failed.";
    return nullptr;
  }
  memset(lsh_project_param, 0, sizeof(LshProjectionParameter));
  lsh_project_param->op_parameter_.type_ = schema::PrimitiveType_LshProjection;

  lsh_project_param->lsh_type_ = lsh_projection_prim->type();
  return reinterpret_cast<OpParameter *>(lsh_project_param);
}
}  // namespace

Registry g_lshProjectionV0ParameterRegistry(schema::v0::PrimitiveType_LshProjection, PopulateLshProjectionParameter,
                                            SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
