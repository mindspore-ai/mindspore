/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/ops/resize.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/resize_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateResizeParameter(const mindspore::lite::PrimitiveC *primitive) {
  ResizeParameter *resize_param = reinterpret_cast<ResizeParameter *>(malloc(sizeof(ResizeParameter)));
  if (resize_param == nullptr) {
    MS_LOG(ERROR) << "malloc ResizeParameter failed.";
    return nullptr;
  }
  memset(resize_param, 0, sizeof(ResizeParameter));
  resize_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Resize *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  resize_param->method_ = static_cast<int>(param->GetMethod());
  resize_param->new_height_ = param->new_height();
  resize_param->new_width_ = param->new_width();
  resize_param->coordinate_transform_mode_ = param->GetCoordinateTransformMode();
  resize_param->preserve_aspect_ratio_ = param->GetPreserveAspectRatio();
  return reinterpret_cast<OpParameter *>(resize_param);
}

Registry ResizeParameterRegistry(schema::PrimitiveType_Resize, PopulateResizeParameter);

}  // namespace lite

}  // namespace mindspore
