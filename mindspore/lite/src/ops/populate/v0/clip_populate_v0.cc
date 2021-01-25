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
#include "nnacl/clip.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateClipParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto clip_prim = primitive->value_as_Clip();
  ClipParameter *act_param = reinterpret_cast<ClipParameter *>(malloc(sizeof(ClipParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ClipParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ClipParameter));
  act_param->op_parameter_.type_ = schema::PrimitiveType_Clip;

  act_param->min_val_ = clip_prim->min();
  act_param->max_val_ = clip_prim->max();
  return reinterpret_cast<OpParameter *>(act_param);
}
}  // namespace

Registry g_clipV0ParameterRegistry(schema::v0::PrimitiveType_Clip, PopulateClipParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
