/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "nnacl/nllloss_parameter.h"
#include "ops/nllloss.h"
#include "ops/grad/nllloss_grad.h"
using mindspore::ops::kNameNLLLoss;
using mindspore::ops::kNameNLLLossGrad;
using mindspore::schema::PrimitiveType_NLLLoss;
using mindspore::schema::PrimitiveType_NLLLossGrad;

namespace mindspore {
namespace lite {
OpParameter *PopulateNLLLossOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<NLLLossParameter *>(PopulateOpParameter<NLLLossParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new NLLLossParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::NLLLoss *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not NLLLoss.";
    return nullptr;
  }
  param->reduction_type_ = static_cast<ReductionType>(op->get_reduction());
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateNLLLossGradOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<NLLLossParameter *>(PopulateOpParameter<NLLLossParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new NLLLossParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::NLLLossGrad *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not NLLLossGrad.";
    return nullptr;
  }
  param->reduction_type_ = static_cast<ReductionType>(op->get_reduction());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameNLLLoss, PrimitiveType_NLLLoss, PopulateNLLLossOpParameter)
REG_OPERATOR_POPULATE(kNameNLLLossGrad, PrimitiveType_NLLLossGrad, PopulateNLLLossGradOpParameter)
}  // namespace lite
}  // namespace mindspore
