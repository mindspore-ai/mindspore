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
#include "nnacl/scatter_elements_parameter.h"
#include "ops/scatter_elements.h"
using mindspore::ops::kNameScatterElements;
using mindspore::schema::PrimitiveType_ScatterElements;

namespace mindspore {
namespace lite {
OpParameter *PopulateScatterElementsOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ScatterElementsParameter *>(PopulateOpParameter<ScatterElementsParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ScatterElementsParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ScatterElements *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not ScatterElements.";
    return nullptr;
  }
  auto axis = op->get_axis();
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = static_cast<int>(axis);
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameScatterElements, PrimitiveType_ScatterElements, PopulateScatterElementsOpParameter);
}  // namespace lite
}  // namespace mindspore
