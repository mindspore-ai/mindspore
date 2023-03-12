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
#include "nnacl/dynamic_quant_parameter.h"
#include "ops/dynamic_quant.h"
using mindspore::ops::kNameDynamicQuant;
using mindspore::schema::PrimitiveType_DynamicQuant;
namespace mindspore {
namespace lite {
OpParameter *PopulateDynamicQuantOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<DynamicQuantParameter *>(PopulateOpParameter<DynamicQuantParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new DynamicQuantParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::DynamicQuant *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to DynamicQuant failed";
    free(param);
    return nullptr;
  }

  param->dst_type_ = op->get_dst_type();
  param->symmetric_ = op->get_symmetric();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameDynamicQuant, PrimitiveType_DynamicQuant, PopulateDynamicQuantOpParameter)
}  // namespace lite
}  // namespace mindspore
