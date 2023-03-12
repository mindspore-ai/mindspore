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
#include "ops/crop_and_resize.h"
using mindspore::ops::kNameCropAndResize;
using mindspore::schema::PrimitiveType_CropAndResize;
namespace mindspore {
namespace lite {
OpParameter *PopulateCropAndResizeOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<CropAndResizeParameter *>(PopulateOpParameter<CropAndResizeParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new CropAndResizeParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::CropAndResize *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to CropAndResize failed";
    free(param);
    return nullptr;
  }
  param->method_ = static_cast<int>(op->get_method());
  param->extrapolation_value_ = op->get_extrapolation_value();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameCropAndResize, PrimitiveType_CropAndResize, PopulateCropAndResizeOpParameter)
}  // namespace lite
}  // namespace mindspore
