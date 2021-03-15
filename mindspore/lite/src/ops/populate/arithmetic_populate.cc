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
#include "src/ops/populate/arithmetic_populate.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
ArithmeticParameter *PopulateArithmeticCommonPara(const void *prim) {
  ArithmeticParameter *param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArithmeticParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  param->op_parameter_.type_ = primitive->value_type();
  param->broadcasting_ = false;
  param->ndim_ = 0;
  param->activation_type_ = 0;
  return param;
}

OpParameter *PopulateArithmetic(const void *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

Registry g_MinimunGradParameterRegistry(schema::PrimitiveType_MinimumGrad, PopulateArithmetic, SCHEMA_CUR);
Registry g_MaximunGradParameterRegistry(schema::PrimitiveType_MaximumGrad, PopulateArithmetic, SCHEMA_CUR);
Registry g_realDivParameterRegistry(schema::PrimitiveType_RealDiv, PopulateArithmetic, SCHEMA_CUR);
Registry g_logicalAndParameterRegistry(schema::PrimitiveType_LogicalAnd, PopulateArithmetic, SCHEMA_CUR);
Registry g_parameterRegistry(schema::PrimitiveType_LogicalOr, PopulateArithmetic, SCHEMA_CUR);
Registry g_equalParameterRegistry(schema::PrimitiveType_Equal, PopulateArithmetic, SCHEMA_CUR);
Registry g_notEqualParameterRegistry(schema::PrimitiveType_NotEqual, PopulateArithmetic, SCHEMA_CUR);
Registry g_essParameterRegistry(schema::PrimitiveType_Less, PopulateArithmetic, SCHEMA_CUR);
Registry g_lessEqualParameterRegistry(schema::PrimitiveType_LessEqual, PopulateArithmetic, SCHEMA_CUR);
Registry g_greaterParameterRegistry(schema::PrimitiveType_Greater, PopulateArithmetic, SCHEMA_CUR);
Registry g_greaterEqualParameterRegistry(schema::PrimitiveType_GreaterEqual, PopulateArithmetic, SCHEMA_CUR);
Registry g_maximumParameterRegistry(schema::PrimitiveType_Maximum, PopulateArithmetic, SCHEMA_CUR);
Registry g_minimumParameterRegistry(schema::PrimitiveType_Minimum, PopulateArithmetic, SCHEMA_CUR);
Registry g_floorDivParameterRegistry(schema::PrimitiveType_FloorDiv, PopulateArithmetic, SCHEMA_CUR);
Registry g_floorModParameterRegistry(schema::PrimitiveType_FloorMod, PopulateArithmetic, SCHEMA_CUR);
Registry g_modParameterRegistry(schema::PrimitiveType_Mod, PopulateArithmetic, SCHEMA_CUR);
Registry g_squaredDifferenceParameterRegistry(schema::PrimitiveType_SquaredDifference, PopulateArithmetic, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
