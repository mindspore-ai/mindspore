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
#include "src/ops/populate/populate_register.h"
#include "src/ops/populate/arithmetic_populate.h"
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore {
namespace lite {
OpParameter *PopulateEltwiseParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Eltwise();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  ArithmeticParameter *param = PopulateArithmeticCommonPara(prim);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }

  param->eltwise_mode_ = value->mode();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Eltwise, PopulateEltwiseParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
