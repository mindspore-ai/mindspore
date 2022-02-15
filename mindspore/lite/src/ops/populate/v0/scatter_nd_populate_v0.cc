/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "nnacl/base/scatter_nd_base.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateScatterNDParameter(const void *prim) {
  auto *param = static_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ScatterNDParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ScatterNDParameter));
  param->op_parameter.type_ = schema::PrimitiveType_ScatterNd;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_scatterNDV0ParameterRegistry(schema::v0::PrimitiveType_ScatterND, PopulateScatterNDParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
