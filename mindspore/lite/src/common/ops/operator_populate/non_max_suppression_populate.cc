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
#include "nnacl/non_max_suppression_parameter.h"
#include "ops/non_max_suppression.h"
using mindspore::ops::kNameNonMaxSuppression;
using mindspore::schema::PrimitiveType_NonMaxSuppression;

namespace mindspore {
namespace lite {
OpParameter *PopulateNonMaxSuppressionOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<NMSParameter *>(PopulateOpParameter<NMSParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new NMSParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::NonMaxSuppression *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PadFusion.";
    return nullptr;
  }

  param->center_point_box_ = static_cast<int>(op->get_center_point_box());
  if (param->center_point_box_ != C0NUM && param->center_point_box_ != C1NUM) {
    MS_LOG(ERROR) << "invalid center_point_box value: " << param->center_point_box_;
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameNonMaxSuppression, PrimitiveType_NonMaxSuppression, PopulateNonMaxSuppressionOpParameter);
}  // namespace lite
}  // namespace mindspore
