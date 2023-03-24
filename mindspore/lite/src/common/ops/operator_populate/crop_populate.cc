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
#include "nnacl/crop_parameter.h"
#include "ops/crop.h"
using mindspore::ops::kNameCrop;
using mindspore::schema::PrimitiveType_Crop;
namespace mindspore {
namespace lite {
OpParameter *PopulateCropOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<CropParameter *>(PopulateOpParameter<CropParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new CropParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::Crop *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to Crop failed";
    free(param);
    return nullptr;
  }

  auto offset = op->get_offsets();
  if (offset.size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "param offset size(" << offset.size() << ") should <= " << COMM_SHAPE_SIZE;
    free(param);
    return nullptr;
  }
  param->offset_size_ = static_cast<int>(offset.size());
  for (size_t i = 0; i < offset.size(); ++i) {
    param->offset_[i] = *(offset.begin() + i);
  }

  auto axis = op->get_axis();
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameCrop, PrimitiveType_Crop, PopulateCropOpParameter)
}  // namespace lite
}  // namespace mindspore
