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

#include "src/ops/add.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/arithmetic.h"
#include "src/ops/populate/arithmetic_populate.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateAddParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->activation_type_ = reinterpret_cast<const mindspore::lite::Add *>(primitive)->GetActivationType();
  return reinterpret_cast<OpParameter *>(param);
}
Registry AddParameterRegistry(schema::PrimitiveType_Add, PopulateAddParameter);

}  // namespace lite
}  // namespace mindspore
