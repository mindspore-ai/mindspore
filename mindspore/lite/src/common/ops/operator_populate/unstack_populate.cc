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
#include "nnacl/unstack_parameter.h"
#include "ops/unstack.h"
using mindspore::ops::kAxis;
using mindspore::ops::kNameUnstack;
using mindspore::schema::PrimitiveType_Unstack;

namespace mindspore {
namespace lite {
OpParameter *PopulateUnstackOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<UnstackParameter *>(PopulateOpParameter<UnstackParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new UnstackParameter failed.";
    return nullptr;
  }

  mindspore::ValuePtr attr = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto axis = GetValue<int64_t>(attr);
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameUnstack, PrimitiveType_Unstack, PopulateUnstackOpParameter)
}  // namespace lite
}  // namespace mindspore
