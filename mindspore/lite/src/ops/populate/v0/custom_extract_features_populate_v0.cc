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

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateExtractFeaturesParameter(const void *prim) {
  OpParameter *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new OpParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(OpParameter));
  auto type = reinterpret_cast<const schema::v0::Primitive *>(prim)->value_type();
  if (type == schema::v0::PrimitiveType_CustomExtractFeatures) {
    param->type_ = schema::PrimitiveType_CustomExtractFeatures;
  } else {
    param->type_ = type;
  }
  return param;
}
}  // namespace

Registry g_customExtractFeaturesV0ParameterRegistry(schema::v0::PrimitiveType_CustomExtractFeatures,
                                                    PopulateExtractFeaturesParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
