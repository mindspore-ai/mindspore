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
#include "src/common/log_adapter.h"
#include "nnacl/reshape_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateReshapeParameter(const void *prim) {
  ReshapeParameter *reshape_param = reinterpret_cast<ReshapeParameter *>(malloc(sizeof(ReshapeParameter)));
  if (reshape_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReshapeParameter failed.";
    return nullptr;
  }
  memset(reshape_param, 0, sizeof(ReshapeParameter));
  reshape_param->op_parameter_.type_ = schema::PrimitiveType_Reshape;
  return reinterpret_cast<OpParameter *>(reshape_param);
}
}  // namespace

Registry g_reshapeV0ParameterRegistry(schema::v0::PrimitiveType_Reshape, PopulateReshapeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
