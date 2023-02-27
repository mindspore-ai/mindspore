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
#include "nnacl/power_parameter.h"
#include "ops/fusion/pow_fusion.h"
using mindspore::ops::kNamePowFusion;
using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulatePowerOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PowerParameter *>(PopulateOpParameter<PowerParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PowerParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::PowFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PriorBox.";
    return nullptr;
  }

  param->scale_ = op->get_scale();
  param->shift_ = op->get_shift();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNamePowFusion, PrimitiveType_PowFusion, PopulatePowerOpParameter)
}  // namespace lite
}  // namespace mindspore
