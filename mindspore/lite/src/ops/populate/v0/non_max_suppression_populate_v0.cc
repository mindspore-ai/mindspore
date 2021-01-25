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
#include "nnacl/non_max_suppression_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateNonMaxSuppressionParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto non_max_suppression_prim = primitive->value_as_NonMaxSuppression();
  NMSParameter *param = reinterpret_cast<NMSParameter *>(malloc(sizeof(NMSParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc param failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(NMSParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_NonMaxSuppression;

  param->center_point_box_ = non_max_suppression_prim->centerPointBox();
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_nonMaxSuppressionV0ParameterRegistry(schema::v0::PrimitiveType_NonMaxSuppression,
                                                PopulateNonMaxSuppressionParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
