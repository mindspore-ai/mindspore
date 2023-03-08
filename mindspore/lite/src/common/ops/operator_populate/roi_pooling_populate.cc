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
#include "nnacl/fp32/roi_pooling_fp32.h"
#include "ops/roi_pooling.h"
using mindspore::ops::kNameROIPooling;
using mindspore::schema::PrimitiveType_ROIPooling;

namespace mindspore {
namespace lite {
OpParameter *PopulateROIPoolingOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ROIPoolingParameter *>(PopulateOpParameter<ROIPoolingParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ROIPoolingParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ROIPooling *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not ROIPooling.";
    return nullptr;
  }

  param->pooledH_ = static_cast<int>(op->get_pooled_h());
  param->pooledW_ = static_cast<int>(op->get_pooled_w());
  param->scale_ = op->get_scale();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameROIPooling, PrimitiveType_ROIPooling, PopulateROIPoolingOpParameter)
}  // namespace lite
}  // namespace mindspore
