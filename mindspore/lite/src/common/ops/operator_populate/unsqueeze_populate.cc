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
#include "nnacl/unsqueeze_parameter.h"
#include "ops/unsqueeze.h"
using mindspore::ops::kAxis;
using mindspore::ops::kNameUnsqueeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore {
namespace lite {
OpParameter *PopulateUnsqueezeOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<UnSqueezeParameter *>(PopulateOpParameter<UnSqueezeParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new UnSqueezeParameter failed.";
    return nullptr;
  }

  ValuePtr attr = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto flat_axis = GetValue<std::vector<int64_t>>(attr);
  if (flat_axis.size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Invalid axis size " << flat_axis.size();
    free(param);
    return nullptr;
  }
  param->num_dim_ = flat_axis.size();
  int i = 0;
  for (auto flat_axi : flat_axis) {
    CHECK_LESS_RETURN_RET(INT32_MAX, flat_axi, nullptr, param);
    param->dims_[i++] = flat_axi;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameUnsqueeze, PrimitiveType_Unsqueeze, PopulateUnsqueezeOpParameter)
}  // namespace lite
}  // namespace mindspore
