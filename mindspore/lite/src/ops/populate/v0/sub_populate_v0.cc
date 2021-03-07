/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/arithmetic.h"
#include "src/ops/populate/arithmetic_populate.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSubParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto sub_prim = primitive->value_as_Sub();
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_SubFusion;  // note: maybe error noneed?
  param->activation_type_ = sub_prim->activationType();
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_subV0ParameterRegistry(schema::v0::PrimitiveType_Sub, PopulateSubParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
