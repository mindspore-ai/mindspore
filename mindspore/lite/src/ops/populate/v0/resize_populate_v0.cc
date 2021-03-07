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
#include "nnacl/resize_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateResizeParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto resize_prim = primitive->value_as_Resize();
  ResizeParameter *resize_param = reinterpret_cast<ResizeParameter *>(malloc(sizeof(ResizeParameter)));
  if (resize_param == nullptr) {
    MS_LOG(ERROR) << "malloc ResizeParameter failed.";
    return nullptr;
  }
  memset(resize_param, 0, sizeof(ResizeParameter));
  resize_param->op_parameter_.type_ = schema::PrimitiveType_Resize;

  resize_param->method_ = static_cast<int>(resize_prim->method());
  resize_param->new_height_ = resize_prim->newHeight();
  resize_param->new_width_ = resize_prim->newWidth();
  if (resize_prim->alignCorners()) {
    resize_param->coordinate_transform_mode_ = 1;
  } else {
    resize_param->coordinate_transform_mode_ = 0;
  }
  resize_param->preserve_aspect_ratio_ = resize_prim->preserveAspectRatio();
  return reinterpret_cast<OpParameter *>(resize_param);
}
}  // namespace

Registry g_resizeV0ParameterRegistry(schema::v0::PrimitiveType_Resize, PopulateResizeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
