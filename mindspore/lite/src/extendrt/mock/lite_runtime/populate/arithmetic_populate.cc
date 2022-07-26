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
#include "extendrt/mock/lite_runtime/populate/arithmetic_populate.h"
#include "extendrt/mock/lite_runtime/populate/base_operator_populate_register.h"
#include "ops/base_operator.h"
#include "ops/primitive_c.h"

using mindspore::schema::PrimitiveType_BiasAddGrad;
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
ArithmeticParameter *PopulateArithmeticCommonPara(void *base_operator) {
  MS_CHECK_TRUE_RET(base_operator != nullptr, nullptr);
  auto base_operator_ptr = static_cast<ops::BaseOperator *>(base_operator);
  if (base_operator_ptr == nullptr) {
    MS_LOG(ERROR) << "cast to BaseOperator failed";
    return nullptr;
  }
  auto *param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArithmeticParameter));

  auto prim_c = base_operator_ptr->GetPrim();
  auto op_type_str = prim_c->instance_name();
  auto type = BaseOperatorPopulateRegistry::GetInstance()->TypeStrToType(op_type_str);
  param->op_parameter_.type_ = type;
  param->broadcasting_ = false;
  param->ndim_ = 0;
  param->activation_type_ = 0;
  return param;
}

OpParameter *PopulateArithmetic(void *base_operator) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(base_operator);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_BASE_POPULATE(PrimitiveType_MinimumGrad, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_MaximumGrad, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_RealDiv, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_LogicalAnd, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_LogicalOr, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_Equal, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_NotEqual, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_Less, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_LessEqual, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_Greater, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_GreaterEqual, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_Maximum, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_Minimum, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_FloorDiv, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_FloorMod, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_Mod, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_SquaredDifference, PopulateArithmetic)
REG_BASE_POPULATE(PrimitiveType_BiasAddGrad, PopulateArithmetic)
}  // namespace mindspore
