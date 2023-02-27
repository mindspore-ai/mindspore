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
#include "nnacl/squeeze_parameter.h"
#include "ops/squeeze.h"
using mindspore::ops::kAxis;
using mindspore::ops::kNameSqueeze;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore {
namespace lite {
OpParameter *PopulateSqueezeOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SqueezeParameter *>(PopulateOpParameter<SqueezeParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SqueezeParameter failed.";
    return nullptr;
  }

  mindspore::ValuePtr attr = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }

  auto flat_axis = GetValue<std::vector<int64_t>>(attr);
  if (flat_axis.size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Invalid axis size " << flat_axis.size();
    free(param);
    return nullptr;
  }

  param->axis_size_ = flat_axis.size();
  for (size_t i = 0; i < param->axis_size_; i++) {
    CHECK_LESS_RETURN_RET(INT32_MAX, flat_axis[i], nullptr, param);
    param->axis_[i] = flat_axis[i];
  }

  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSqueeze, PrimitiveType_Squeeze, PopulateSqueezeOpParameter)
}  // namespace lite
}  // namespace mindspore
