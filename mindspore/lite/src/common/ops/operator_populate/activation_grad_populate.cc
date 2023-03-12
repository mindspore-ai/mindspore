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
#include "nnacl/fp32_grad/activation_grad_fp32.h"
#include "ops/grad/activation_grad.h"
using mindspore::ops::kNameActivationGrad;
using mindspore::schema::PrimitiveType_ActivationGrad;
namespace mindspore {
namespace lite {
OpParameter *PopulateActivationGradOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ActivationGradParameter *>(PopulateOpParameter<ActivationGradParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ActivationGradParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ActivationGrad *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to ActivationGrad failed";
    free(param);
    return nullptr;
  }
  param->type_ = static_cast<int>(op->get_activation_type());
  param->alpha_ = op->get_alpha();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameActivationGrad, PrimitiveType_ActivationGrad, PopulateActivationGradOpParameter)
}  // namespace lite
}  // namespace mindspore
