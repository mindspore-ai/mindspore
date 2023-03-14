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
#include "nnacl/pad_parameter.h"
#include "ops/fusion/pad_fusion.h"
using mindspore::ops::kNamePadFusion;
using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulatePadOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PadParameter *>(PopulateOpParameter<PadParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PadParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::PadFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PadFusion.";
    return nullptr;
  }

  param->pad_mode_ = static_cast<int>(op->get_padding_mode());
  param->constant_value_ = op->get_constant_value();
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNamePadFusion, PrimitiveType_PadFusion, PopulatePadOpParameter)
}  // namespace lite
}  // namespace mindspore
