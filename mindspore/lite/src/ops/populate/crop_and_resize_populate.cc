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
#include "src/ops/populate/populate_register.h"
#include "nnacl/resize_parameter.h"
using mindspore::schema::PrimitiveType_CropAndResize;

namespace mindspore {
namespace lite {
OpParameter *PopulateCropAndResizeParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_CropAndResize();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<CropAndResizeParameter *>(malloc(sizeof(CropAndResizeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc CropAndResizeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(CropAndResizeParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->method_ = static_cast<int>(value->method());
  param->extrapolation_value_ = value->extrapolation_value();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_CropAndResize, PopulateCropAndResizeParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
