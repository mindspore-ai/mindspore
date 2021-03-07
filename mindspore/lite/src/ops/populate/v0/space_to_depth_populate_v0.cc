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
#include "nnacl/space_to_depth_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSpaceToDepthParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto space_to_depth_prim = primitive->value_as_SpaceToDepth();
  SpaceToDepthParameter *space_depth_param =
    reinterpret_cast<SpaceToDepthParameter *>(malloc(sizeof(SpaceToDepthParameter)));
  if (space_depth_param == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToDepthParameter failed.";
    return nullptr;
  }
  memset(space_depth_param, 0, sizeof(SpaceToDepthParameter));
  space_depth_param->op_parameter_.type_ = schema::PrimitiveType_SpaceToDepth;
  space_depth_param->block_size_ = space_to_depth_prim->blockSize();
  if (space_to_depth_prim->format() != schema::v0::Format::Format_NHWC) {
    MS_LOG(ERROR) << "Currently only NHWC format is supported.";
    free(space_depth_param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(space_depth_param);
}
}  // namespace

Registry g_spaceToDepthV0ParameterRegistry(schema::v0::PrimitiveType_SpaceToDepth, PopulateSpaceToDepthParameter,
                                           SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
