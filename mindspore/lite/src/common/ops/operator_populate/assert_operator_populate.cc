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
#include "ops/assert.h"

using mindspore::ops::kNameAssert;
namespace mindspore {
namespace lite {
OpParameter *PopulateAssertOpParameter(const BaseOperatorPtr &base_operator) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr";
    return nullptr;
  }
  auto op_parameter_ptr = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (op_parameter_ptr == nullptr) {
    MS_LOG(ERROR) << "Make OpParameter ptr failed";
    return nullptr;
  }
  memset(op_parameter_ptr, 0, sizeof(OpParameter));
  auto name = base_operator->name();

  auto iter = kOpNameWithPrimitiveType.find(name);
  if (iter == kOpNameWithPrimitiveType.end()) {
    MS_LOG(ERROR) << "Can not find ParameterPtrGen : " << name;
    return nullptr;
  }
  op_parameter_ptr->type_ = iter->second;
  return op_parameter_ptr;
}

REG_OPERATOR_POPULATE(kNameAssert, PopulateAssertOpParameter)
}  // namespace lite
}  // namespace mindspore
