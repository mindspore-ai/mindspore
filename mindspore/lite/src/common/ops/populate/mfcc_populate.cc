/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/infer/mfcc_infer.h"
using mindspore::schema::PrimitiveType_Mfcc;

namespace mindspore {
namespace lite {
OpParameter *PopulateMfccParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_CHECK_TRUE_MSG(primitive != nullptr, nullptr, "Mfcc primitive is nullptr!");
  auto value = primitive->value_as_Mfcc();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<MfccParameter *>(malloc(sizeof(MfccParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc MfccParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(MfccParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->dct_coeff_num_ = value->dct_coeff_num();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Mfcc, PopulateMfccParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
