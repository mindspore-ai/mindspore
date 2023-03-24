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
#include "nnacl/infer/mfcc_infer.h"
#include "ops/mfcc.h"
using mindspore::ops::kNameMfcc;
using mindspore::schema::PrimitiveType_Mfcc;
namespace mindspore {
namespace lite {
OpParameter *PopulateMfccOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<MfccParameter *>(PopulateOpParameter<MfccParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new MfccParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Mfcc *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to Mfcc failed";
    free(param);
    return nullptr;
  }
  param->dct_coeff_num_ = static_cast<int>(op->get_dct_coeff_num());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameMfcc, PrimitiveType_Mfcc, PopulateMfccOpParameter)
}  // namespace lite
}  // namespace mindspore
