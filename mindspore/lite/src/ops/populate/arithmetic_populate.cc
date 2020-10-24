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
#include "src/ops/arithmetic.h"
#include "src/ops/add.h"
#include "src/ops/sub.h"
#include "src/ops/mul.h"
#include "src/ops/div.h"
#include "src/ops/eltwise.h"
#include "src/ops/greater_equal.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
ArithmeticParameter *PopulateArithmeticCommonPara(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArithmeticParameter));
  param->op_parameter_.type_ = primitive->Type();
  param->broadcasting_ = reinterpret_cast<const lite::Arithmetic *>(primitive)->Broadcasting();
  param->ndim_ = reinterpret_cast<const lite::Arithmetic *>(primitive)->NDims();
  param->activation_type_ = 0;

  auto tmp_shape = reinterpret_cast<const lite::Arithmetic *>(primitive)->InShape0();
  memcpy(param->in_shape0_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = reinterpret_cast<const lite::Arithmetic *>(primitive)->InShape1();
  memcpy(param->in_shape1_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = reinterpret_cast<const lite::Arithmetic *>(primitive)->OutputShape();
  memcpy(param->out_shape_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  return param;
}

OpParameter *PopulateArithmetic(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateAddParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->activation_type_ = reinterpret_cast<const mindspore::lite::Add *>(primitive)->GetActivationType();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateSubParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->activation_type_ = reinterpret_cast<const mindspore::lite::Sub *>(primitive)->GetActivationType();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateMulParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->activation_type_ = reinterpret_cast<const mindspore::lite::Mul *>(primitive)->GetActivationType();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateDivParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->activation_type_ = reinterpret_cast<const mindspore::lite::Div *>(primitive)->GetActivationType();
  return reinterpret_cast<OpParameter *>(param);
}

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

Registry RealDivParameterRegistry(schema::PrimitiveType_RealDiv, PopulateArithmetic);
Registry LogicalAndParameterRegistry(schema::PrimitiveType_LogicalAnd, PopulateArithmetic);
Registry ParameterRegistry(schema::PrimitiveType_LogicalOr, PopulateArithmetic);
Registry EqualParameterRegistry(schema::PrimitiveType_Equal, PopulateArithmetic);
Registry LessParameterRegistry(schema::PrimitiveType_Less, PopulateArithmetic);
Registry GreaterParameterRegistry(schema::PrimitiveType_Greater, PopulateArithmetic);
Registry GreaterEqualParameterRegistry(schema::PrimitiveType_GreaterEqual, PopulateArithmetic);
Registry NotEqualParameterRegistry(schema::PrimitiveType_NotEqual, PopulateArithmetic);
Registry LessEqualParameterRegistry(schema::PrimitiveType_LessEqual, PopulateArithmetic);
Registry MaximumParameterRegistry(schema::PrimitiveType_Maximum, PopulateArithmetic);
Registry MinimumParameterRegistry(schema::PrimitiveType_Minimum, PopulateArithmetic);
Registry FloorDivParameterRegistry(schema::PrimitiveType_FloorDiv, PopulateArithmetic);
Registry FloorModParameterRegistry(schema::PrimitiveType_FloorMod, PopulateArithmetic);
Registry SquaredDifferenceParameterRegistry(schema::PrimitiveType_SquaredDifference, PopulateArithmetic);
Registry AddParameterRegistry(schema::PrimitiveType_Add, PopulateAddParameter);
Registry SubParameterRegistry(schema::PrimitiveType_Sub, PopulateSubParameter);
Registry MulParameterRegistry(schema::PrimitiveType_Mul, PopulateMulParameter);
Registry DivParameterRegistry(schema::PrimitiveType_Div, PopulateDivParameter);
Registry EltwiseParameterRegistry(schema::PrimitiveType_Eltwise, PopulateEltwiseParameter);
}  // namespace lite
}  // namespace mindspore
