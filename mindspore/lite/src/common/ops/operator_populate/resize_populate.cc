/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/resize_parameter.h"
#include "ops/resize.h"
using mindspore::ops::kNameResize;
using mindspore::schema::PrimitiveType_Resize;

namespace mindspore {
namespace lite {
OpParameter *PopulateResizeOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ResizeParameter *>(PopulateOpParameter<ResizeParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ResizeParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Resize *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not Resize.";
    return nullptr;
  }

  param->method_ = static_cast<int>(op->get_method());
  param->new_height_ = op->get_new_height();
  param->new_width_ = op->get_new_width();
  param->coordinate_transform_mode_ = static_cast<int>(op->get_coordinate_transform_mode());
  param->preserve_aspect_ratio_ = op->get_preserve_aspect_ratio();
  param->cubic_coeff_ = op->get_cubic_coeff();
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameResize, PrimitiveType_Resize, PopulateResizeOpParameter)
}  // namespace lite
}  // namespace mindspore
