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
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_MaximumGrad;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_MinimumGrad;
using mindspore::schema::PrimitiveType_Mod;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;

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

REG_POPULATE(PrimitiveType_MinimumGrad, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_MaximumGrad, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_RealDiv, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_LogicalAnd, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_LogicalOr, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Equal, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_NotEqual, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Less, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_LessEqual, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Greater, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_GreaterEqual, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Maximum, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Minimum, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_FloorDiv, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_FloorMod, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Mod, PopulateArithmetic, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_SquaredDifference, PopulateArithmetic, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
