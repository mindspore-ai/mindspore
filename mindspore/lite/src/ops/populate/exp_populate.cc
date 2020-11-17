/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/exp.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/exp_fp32.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateExpParameter(const mindspore::lite::PrimitiveC *primitive) {
  ExpParameter *exp_parameter = reinterpret_cast<ExpParameter *>(malloc(sizeof(ExpParameter)));
  if (exp_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc ExpParameter failed.";
    return nullptr;
  }
  memset(exp_parameter, 0, sizeof(ExpParameter));
  exp_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Exp *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  exp_parameter->base_ = param->GetBase();
  exp_parameter->scale_ = param->GetScale();
  exp_parameter->shift_ = param->GetShift();
  if (exp_parameter->base_ != -1 && exp_parameter->base_ <= 0) {
    MS_LOG(ERROR) << "Exp base must be strictly positive, got " << exp_parameter->base_;
    free(exp_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(exp_parameter);
}

Registry ExpParameterRegistry(schema::PrimitiveType_Exp, PopulateExpParameter);
}  // namespace lite
}  // namespace mindspore
