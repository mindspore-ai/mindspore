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

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateShapeParameter(const void *prim) {
  OpParameter *shape_param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (shape_param == nullptr) {
    MS_LOG(ERROR) << "malloc ShapeParameter failed.";
    return nullptr;
  }
  memset(shape_param, 0, sizeof(OpParameter));
  shape_param->type_ = schema::PrimitiveType_Shape;
  return reinterpret_cast<OpParameter *>(shape_param);
}
}  // namespace

Registry g_shapeV0ParameterRegistry(schema::v0::PrimitiveType_Shape, PopulateShapeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
