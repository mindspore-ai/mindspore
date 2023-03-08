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
#include "nnacl/fp32/reverse_fp32.h"
#include "ops/reverse_v2.h"
using mindspore::ops::kNameReverseV2;
using mindspore::schema::PrimitiveType_ReverseV2;

namespace mindspore {
namespace lite {
OpParameter *PopulateReverseOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ReverseParameter *>(PopulateOpParameter<ReverseParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ReverseV2 *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not ReverseV2.";
    return nullptr;
  }

  auto flatAxis = op->get_axis();
  param->num_axis_ = static_cast<int>(flatAxis.size());
  if (param->num_axis_ > REVERSE_SHAPE_MAX_SIZE) {
    MS_LOG(ERROR) << "Invalid axis size: " << param->num_axis_;
    free(param);
    return nullptr;
  }
  int i = 0;
  for (auto flatAxi : flatAxis) {
    param->axis_[i++] = static_cast<int>(flatAxi);
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameReverseV2, PrimitiveType_ReverseV2, PopulateReverseOpParameter)
}  // namespace lite
}  // namespace mindspore
