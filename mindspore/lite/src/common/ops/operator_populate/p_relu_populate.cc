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
#include "nnacl/prelu_parameter.h"
#include "ops/fusion/prelu_fusion.h"
using mindspore::ops::kNamePReLUFusion;
using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulatePReLUOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PReluParameter *>(PopulateOpParameter<PReluParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PReluParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::PReLUFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PadFusion.";
    return nullptr;
  }

  param->channelShared = op->get_channel_shared();
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNamePReLUFusion, PrimitiveType_PReLUFusion, PopulatePReLUOpParameter)
}  // namespace lite
}  // namespace mindspore
