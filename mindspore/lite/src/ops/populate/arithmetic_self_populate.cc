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
#include "src/common/log_adapter.h"
#include "nnacl/arithmetic_self_parameter.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateArithmeticSelf(const void *prim) {
  ArithmeticSelfParameter *arithmetic_self_param =
    reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticSelfParameter failed.";
    return nullptr;
  }
  memset(arithmetic_self_param, 0, sizeof(ArithmeticSelfParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  arithmetic_self_param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}

Registry g_absParameterRegistry(schema::PrimitiveType_Abs, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_cosParameterRegistry(schema::PrimitiveType_Cos, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_sinParameterRegistry(schema::PrimitiveType_Sin, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_logParameterRegistry(schema::PrimitiveType_Log, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_negParameterRegistry(schema::PrimitiveType_Neg, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_negGradParameterRegistry(schema::PrimitiveType_NegGrad, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_logGradParameterRegistry(schema::PrimitiveType_LogGrad, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_sqrtParameterRegistry(schema::PrimitiveType_Sqrt, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_squareParameterRegistry(schema::PrimitiveType_Square, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_rsqrtParameterRegistry(schema::PrimitiveType_Rsqrt, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_logicalNotParameterRegistry(schema::PrimitiveType_LogicalNot, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_floorParameterRegistry(schema::PrimitiveType_Floor, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_ceilParameterRegistry(schema::PrimitiveType_Ceil, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_roundParameterRegistry(schema::PrimitiveType_Round, PopulateArithmeticSelf, SCHEMA_CUR);
Registry g_reciprocalParameterRegistry(schema::PrimitiveType_Reciprocal, PopulateArithmeticSelf, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
