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
#include "nnacl/arithmetic.h"
#include "src/ops/populate/arithmetic_populate.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSubParameter(const void *prim) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(prim);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  auto primitive = static_cast<const schema::Primitive *>(prim);
  param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(param);
}

Registry g_subParameterRegistry(schema::PrimitiveType_SubFusion, PopulateSubParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
