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
#include "nnacl/pad_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulatePadParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto pad_prim = primitive->value_as_Pad();
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "malloc PadParameter failed.";
    return nullptr;
  }
  memset(pad_param, 0, sizeof(PadParameter));
  pad_param->op_parameter_.type_ = schema::PrimitiveType_PadFusion;

  pad_param->pad_mode_ = pad_prim->paddingMode();
  pad_param->constant_value_ = pad_prim->constantValue();
  return reinterpret_cast<OpParameter *>(pad_param);
}
}  // namespace

Registry g_padV0ParameterRegistry(schema::v0::PrimitiveType_Pad, PopulatePadParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
