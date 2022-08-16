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
#include "nnacl/arithmetic.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/ops/populate/arithmetic_populate.h"
using mindspore::schema::PrimitiveType_MulFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateMulParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto mul_param = primitive->value_as_MulFusion();
  if (mul_param == nullptr) {
    MS_LOG(ERROR) << "MulFusion param is nullptr!";
    return nullptr;
  }

  ArithmeticParameter *param = PopulateArithmeticCommonPara(prim);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->activation_type_ = mul_param->activation_type();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_MulFusion, PopulateMulParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
