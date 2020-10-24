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

#include "src/ops/div.h"
#include "src/ops/eltwise.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "src/ops/populate/arithmetic_populate.h"
namespace mindspore {
namespace lite {

OpParameter *PopulateEltwiseParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  auto eltwise = reinterpret_cast<const mindspore::lite::Eltwise *>(primitive);
  switch (eltwise->GetMode()) {
    case schema::EltwiseMode_PROD:
      param->op_parameter_.type_ = schema::PrimitiveType_Mul;
      break;
    case schema::EltwiseMode_SUM:
      param->op_parameter_.type_ = schema::PrimitiveType_Add;
      break;
    case schema::EltwiseMode_MAXIMUM:
      param->op_parameter_.type_ = schema::PrimitiveType_Maximum;
      break;
    default:
      free(param);
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

Registry EltwiseParameterRegistry(schema::PrimitiveType_Eltwise, PopulateEltwiseParameter);

}  // namespace lite
}  // namespace mindspore
