/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/arithmetic_self_parameter.h"
#include "ops/abs.h"
#include "ops/cos.h"
#include "ops/sin.h"
#include "ops/log.h"
#include "ops/grad/log_grad.h"
#include "ops/neg.h"
#include "ops/grad/neg_grad.h"
#include "ops/sqrt.h"
#include "ops/square.h"
#include "ops/rsqrt.h"
#include "ops/logical_not.h"
#include "ops/floor.h"
#include "ops/ceil.h"
#include "ops/round.h"
#include "ops/reciprocal.h"
using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogGrad;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Neg;
using mindspore::schema::PrimitiveType_NegGrad;
using mindspore::schema::PrimitiveType_Reciprocal;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

using mindspore::ops::kNameAbs;
using mindspore::ops::kNameCeil;
using mindspore::ops::kNameCos;
using mindspore::ops::kNameFloor;
using mindspore::ops::kNameLog;
using mindspore::ops::kNameLogGrad;
using mindspore::ops::kNameLogicalNot;
using mindspore::ops::kNameNeg;
using mindspore::ops::kNameNegGrad;
using mindspore::ops::kNameReciprocal;
using mindspore::ops::kNameRound;
using mindspore::ops::kNameRsqrt;
using mindspore::ops::kNameSin;
using mindspore::ops::kNameSqrt;
using mindspore::ops::kNameSquare;

namespace mindspore {
namespace lite {
OpParameter *PopulateArithmeticSelfOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ArithmeticSelfParameter *>(PopulateOpParameter<ArithmeticSelfParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticSelfParameter failed.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameAbs, PrimitiveType_Abs, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameCeil, PrimitiveType_Ceil, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameCos, PrimitiveType_Cos, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameFloor, PrimitiveType_Floor, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameLog, PrimitiveType_Log, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameLogGrad, PrimitiveType_LogGrad, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameLogicalNot, PrimitiveType_LogicalNot, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameNeg, PrimitiveType_Neg, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameNegGrad, PrimitiveType_NegGrad, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameReciprocal, PrimitiveType_Reciprocal, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameRound, PrimitiveType_Round, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameRsqrt, PrimitiveType_Rsqrt, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameSin, PrimitiveType_Sin, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameSqrt, PrimitiveType_Sqrt, PopulateArithmeticSelfOpParameter)
REG_OPERATOR_POPULATE(kNameSquare, PrimitiveType_Square, PopulateArithmeticSelfOpParameter)
}  // namespace lite
}  // namespace mindspore
